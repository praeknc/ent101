#!/usr/bin/env python3
"""
scrape_ai_pricing.py  -  AI API token-price panel builder (for the TME thesis)

TWO MODES
  live     Snapshot today's official pricing pages, save the raw page, parse
           price tables into prices.csv (appends, so re-running builds a panel).
  history  Pull archived versions of the same pages from the Wayback Machine
           (monthly snapshots), save raw HTML, and extract candidate prices.

SETUP
  pip install requests beautifulsoup4 lxml playwright
  playwright install chromium        # needed for JS-rendered pages (Alibaba, Baidu)

USAGE
  python scrape_ai_pricing.py live
  python scrape_ai_pricing.py live --provider openai anthropic
  python scrape_ai_pricing.py history --since 2023-01 --until 2026-06
  python scrape_ai_pricing.py history --provider openai --limit 12
  python scrape_ai_pricing.py reparse        # re-extract from saved pages, offline

OUTPUT (./pricing_data/)
  raw/<provider>/<timestamp>.html|md   untouched page as fetched
  raw/<provider>/<timestamp>.png       screenshot (rendered pages only)
  manifest.jsonl                       url, sha256, time, method for every fetch
  prices.csv                           parsed LIVE rows (append-only)
  prices_history.csv                   parsed WAYBACK rows (append-only)
  history_candidates.csv               raw text lines that look like prices

IMPORTANT
  * Parsers are heuristic. Anything with needs_review=True must be checked
    against the saved raw page / screenshot before it goes in the thesis.
  * Prices are normalised to "per 1M tokens". Per-1K prices are x1000.
  * RMB and USD are NOT converted here. Keep the currency column.
  * Check each site's terms/robots before heavy use. The script is rate-limited.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT = Path("pricing_data")
RAW = OUT / "raw"
LIVE_CSV = OUT / "prices.csv"
HIST_CSV = OUT / "prices_history.csv"
CAND_CSV = OUT / "history_candidates.csv"
MANIFEST = OUT / "manifest.jsonl"

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36 academic-pricing-research")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept-Language": "en,zh-CN;q=0.8"})
POLITE_DELAY = 1.5  # seconds between requests
WAYBACK_DELAY = 5.0  # web.archive.org allows ~15 req/min; faster gets 429s and a temporary block

# --------------------------------------------------------------------------
# SOURCES  (edit freely; verify the URLs in a browser first)
#   authoritative: official first-party page -> fewer review flags
#   render: "no" = plain HTTP, "yes" = Playwright, "auto" = HTTP then Playwright
#   parser: "generic" (any table with Input/Output columns) or "gemini"
# --------------------------------------------------------------------------
SOURCES = [
    dict(provider="openai", url="https://developers.openai.com/api/docs/pricing.md",
         render="no", parser="generic", authoritative=True,
         default_unit="1M"),  # page header states "Prices per 1M tokens"
    dict(provider="anthropic", url="https://platform.claude.com/docs/en/about-claude/pricing",
         render="auto", parser="generic", authoritative=True),
    dict(provider="google", url="https://ai.google.dev/gemini-api/docs/pricing",
         render="auto", parser="gemini", authoritative=True),
    # JS-rendered, must use a browser. Confirm these URLs yourself.
    dict(provider="alibaba", url="https://www.alibabacloud.com/help/en/model-studio/pricing",
         render="yes", parser="generic", authoritative=True),
    dict(provider="baidu", url="https://intl.cloud.baidu.com/zh/doc/qianfan/s/Jm8r1826a-intl",
         render="yes", parser="long", authoritative=True),  # one row per 输入/输出 sub-item
]

# URLs to look up on the Wayback Machine. Old pricing pages moved around, so
# several URLs per provider. Unknown/dead URLs are simply reported as "0 snapshots".
HISTORY = {
    "openai": ["https://openai.com/api/pricing/", "https://openai.com/pricing",
               "https://platform.openai.com/docs/pricing",
               "https://developers.openai.com/api/docs/pricing"],
    "anthropic": ["https://www.anthropic.com/pricing", "https://www.anthropic.com/api",
                  "https://docs.anthropic.com/en/docs/about-claude/models",
                  "https://docs.anthropic.com/en/docs/about-claude/pricing",
                  "https://platform.claude.com/docs/en/about-claude/pricing"],
    "google": ["https://ai.google.dev/pricing", "https://ai.google.dev/gemini-api/docs/pricing"],
    "alibaba": ["https://www.alibabacloud.com/help/en/model-studio/pricing",
                "https://help.aliyun.com/zh/model-studio/billing-for-model-studio"],
    "baidu": ["https://intl.cloud.baidu.com/zh/doc/qianfan/s/Jm8r1826a-intl",
              "https://cloud.baidu.com/doc/WENXINWORKSHOP/s/hlrk4akp7"],
}

FIELDS = ["provider", "model", "section", "input_per_1m", "output_per_1m", "currency",
          "unit_detected", "needs_review", "review_reason", "raw_row", "source_url",
          "source_type", "retrieved_at", "snapshot_sha256"]

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
PRICE_RE = re.compile(
    r'(?:US\$|\$|¥|￥|RMB\s?)\s*([0-9][0-9,]*(?:\.[0-9]+)?)'
    r'|([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:元|RMB|CNY)')


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_price(text):
    """Return (value, currency, unit) for the first price in text, else None."""
    m = PRICE_RE.search(text or "")
    if not m:
        return None
    val = float((m.group(1) or m.group(2)).replace(",", ""))
    cur = "CNY" if re.search(r'[¥￥]|RMB|CNY|元', text) else "USD"
    t = text.lower()
    if re.search(r'1m|mtok|million|百万', t):
        unit = "1M"
    elif re.search(r'1k|千|per 1,?000', t):
        unit = "1K"
    else:
        unit = ""
    return val, cur, unit


def per_1m(val, unit):
    return val * 1000 if unit == "1K" else val


class RateLimited(RuntimeError):
    pass


def http_get(url, params=None, tries=4, timeout=60, backoff=5):
    """GET with retries. 429/502/503/504 back off (honouring Retry-After);
    the final error always names the last HTTP status or exception."""
    last = None
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 502, 503, 504):
                ra = r.headers.get("Retry-After", "")
                wait = int(ra) if ra.isdigit() else backoff * 2 ** i
                if i < tries - 1:
                    print(f"      {last}; waiting {wait}s")
                    time.sleep(wait)
                continue
            r.raise_for_status()
            break  # other non-200 2xx/3xx: not retryable
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"GET failed for {url}: {e}")  # 4xx: retrying won't help
        except requests.exceptions.ProxyError as e:
            raise RuntimeError(f"GET blocked by proxy for {url}: {e}")  # retrying won't help
        except requests.RequestException as e:
            last = e
            time.sleep(3 * (i + 1))
    err = RateLimited if last == "HTTP 429" else RuntimeError
    raise err(f"GET failed for {url}: {last}")


CHROMIUM_FALLBACKS = [os.environ.get("CHROMIUM_PATH", ""), "/opt/pw-browsers/chromium",
                      "/usr/bin/chromium", "/usr/bin/chromium-browser",
                      "/usr/bin/google-chrome"]


def _launch_chromium(p):
    """Launch Playwright's bundled chromium; if that build is not installed
    (pip playwright newer than the downloaded browser), use a system chromium."""
    try:
        return p.chromium.launch()
    except Exception as e:
        if "Executable doesn't exist" not in str(e):
            raise
        for exe in CHROMIUM_FALLBACKS:
            if exe and Path(exe).exists():
                return p.chromium.launch(executable_path=exe,
                                         args=["--headless=new", "--no-sandbox"])
        raise


def fetch_rendered(url, shot_path):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("playwright not installed (pip install playwright; "
                           "playwright install chromium)")
    with sync_playwright() as p:
        browser = _launch_chromium(p)
        page = browser.new_page(user_agent=UA, viewport={"width": 1400, "height": 2000})
        page.goto(url, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(3000)
        html = page.content()
        try:
            page.screenshot(path=str(shot_path), full_page=True)
        except Exception:
            pass
        browser.close()
    return html


def save_snapshot(provider, url, content, ext, method, stamp=None):
    stamp = stamp or now_utc()
    d = RAW / provider
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stamp}.{ext}"
    data = content.encode("utf-8", errors="replace")
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(provider=provider, url=url, file=str(path), sha256=sha,
                                bytes=len(data), method=method,
                                fetched_at=now_utc())) + "\n")
    return path, sha


def append_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# block extraction: turn HTML or Markdown into a stream of
#   ("heading", text) and ("table", [[cell, ...], ...])
# --------------------------------------------------------------------------
def html_blocks(html):
    soup = BeautifulSoup(html, "lxml")
    for el in soup.find_all(["h1", "h2", "h3", "h4", "table"]):
        if el.name == "table":
            rows = [[c.get_text("\n", strip=True) for c in tr.find_all(["th", "td"])]
                    for tr in el.find_all("tr")]
            yield "table", [r for r in rows if any(r)]
        else:
            # strip icon-font glyphs (Unicode private-use area) from headings
            yield "heading", re.sub(r'[\ue000-\uf8ff]', '', el.get_text(" ", strip=True)).strip()


TIER_LINE = re.compile(r'^(?:#+\s*)?(Standard|Batch|Flex|Fast|Ultrafast|Priority)'
                       r'(?:\s+pricing data)?\s*$', re.I)


GENERIC_HEADING = re.compile(r'^#+\s*(grouped\s+)?(\w+\s+)?pricing table data$|^#+\s*\w+\s+pricing data$', re.I)
TOPIC_LINE = re.compile(r'^[A-Z][A-Za-z0-9 &/+()\-]{1,40}$')
NOT_TOPIC = {"Speed"}


def md_blocks(text):
    """Markdown -> blocks. Besides '#' headings, yields ("tier", Standard/Batch/...)
    and ("topic", ...) for the bare title lines OpenAI's pricing.md uses
    ("Images", "Realtime & Audio", "Tools"...)."""
    table = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r':?-{2,}:?', c) for c in cells if c):
                continue  # separator row
            table.append(cells)
            continue
        if table:
            yield "table", table
            table = []
        m = TIER_LINE.match(s)
        if m:
            yield "tier", m.group(1).title()
        elif GENERIC_HEADING.match(s):
            continue  # "### Grouped Pricing Table data" etc. carry no information
        elif s.startswith("#"):
            yield "heading", s.lstrip("# ").strip()
        elif TOPIC_LINE.match(s) and s not in NOT_TOPIC:
            yield "topic", s
    if table:
        yield "table", table


def blocks_for(content):
    low = content[:5000].lower()
    if "<html" in low or "<table" in content.lower() or "<!doctype" in low:
        return html_blocks(content)
    return md_blocks(content)


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------
# column synonyms: 2023-era OpenAI pages used "Prompt" / "Completion"
COL_KEYS = {"input": ("input", "prompt"), "output": ("output", "completion"),
            "model": ("model",), "modality": ("modality",)}


def _is_col(h, key):
    return any(k in h for k in COL_KEYS[key])


def _pick(header, key, exclude=("cached", "cache", "long", "write", "batch")):
    if not header:
        return None
    for i, h in enumerate(header):
        if _is_col(h, key) and not any(x in h for x in exclude):
            return i
    return None


PROMPT_TIER_RE = re.compile(r'prompts?\s+(up to|over|above|below|<=?|>=?|≤|≥)\s*([0-9,]+[kK]?)')
ARITH_RE = re.compile(r'×|[0-9]\s*\*\s*\$|/\s*1,?000,?000\b')  # worked-example arithmetic


def parse_generic(content, ctx):
    recs, section, header, last_name = [], "", None, ""
    topic, tier = "", ""
    for kind, payload in blocks_for(content):
        if kind in ("heading", "topic"):
            topic, tier = payload, ""
        elif kind == "tier":
            tier = payload
        if kind != "table":
            section = " / ".join(x for x in (topic, tier) if x)
            continue
        header = None
        other_header = False  # table has a header, but not an Input/Output one
        for cells in payload:
            low = [c.lower() for c in cells]
            has_price = any(parse_price(c) for c in cells)
            if not has_price and any(_is_col(c, "input") for c in low) \
                    and any(_is_col(c, "output") for c in low):
                header, other_header = low, False
                continue
            if not has_price and header is None and len(cells) > 1:
                other_header = True  # e.g. "Line item | Calculation | Cost"
                continue
            if not has_price or not cells:
                continue
            if header is None and other_header:
                continue  # not a per-token rate table (worked examples, tool fees...)
            if any(ARITH_RE.search(c) for c in cells):
                continue
            reasons = []
            mcol = _pick(header, "model", exclude=()) or 0  # "Category | Model | ..." tables
            name = cells[mcol].split("\n")[0].strip() if mcol < len(cells) else ""
            if not name or parse_price(name):
                name = (last_name + " [continuation]") if last_name else ""
                # rowspan'd name cell: row is missing leading cells -> realign
                if header and len(cells) < len(header) and parse_price(cells[0]):
                    cells = [""] * (len(header) - len(cells)) + cells
                    reasons.append("rowspan row realigned to header")
            else:
                last_name = name
            ci, co = _pick(header, "input"), _pick(header, "output")
            if ci is None or co is None or max(ci, co) >= len(cells):
                prices = [c for c in cells[1:] if parse_price(c)]
                if len(prices) < 2:
                    continue
                pin, pout = parse_price(prices[0]), parse_price(prices[1])
                reasons.append("no Input/Output header; used first two price cells")
            else:
                pin, pout = parse_price(cells[ci]), parse_price(cells[co])
            if not pin or not pout:
                continue
            unit = pin[2] or pout[2] or ctx.get("default_unit", "")
            if not unit:
                reasons.append("per-unit not stated in cell; assumed 1M")
            if pin[1] != pout[1]:
                reasons.append("currency mismatch")
            if not ctx["authoritative"]:
                reasons.append("non-authoritative source")
            if "[continuation]" in name and not reasons:
                reasons.append("blank model name (rowspan?)")
            m = PROMPT_TIER_RE.search(" ".join(cells))
            if m:
                name = (name.replace(" [continuation]", "")
                        + f" (prompt {m.group(1)} {m.group(2)} tokens)")
            qcol = _pick(header, "modality", exclude=())
            if qcol is not None and qcol < len(cells) and cells[qcol]:
                name += f" [{cells[qcol]}]"
            recs.append(dict(
                provider=ctx["provider"], model=name, section=section,
                input_per_1m=per_1m(pin[0], unit), output_per_1m=per_1m(pout[0], unit),
                currency=pin[1], unit_detected=unit,
                needs_review=bool(reasons), review_reason="; ".join(reasons),
                raw_row=" | ".join(c.replace("\n", " ") for c in cells[:6])))
    return recs


def parse_gemini(content, ctx):
    """Gemini page: one h2 per model, h3 per tier, tables of 'Input price'/'Output price' rows."""
    recs, model, tier = [], "", ""
    acc = {}
    for kind, payload in html_blocks(content):
        if kind == "heading":
            # h2 = model, h3 = tier; html_blocks flattens both, so use a tier whitelist
            if TIER_LINE.match(payload):
                tier = payload.title()
            else:
                model, tier = payload, ""
            continue
        for cells in payload:
            if len(cells) < 2:
                continue
            label = cells[0].lower()
            if label.startswith("input price") or label.startswith("output price"):
                paid = cells[-1]
                p = parse_price(paid)
                if not p:
                    continue
                key = (model, tier)
                d = acc.setdefault(key, {})
                d["in" if label.startswith("input") else "out"] = (p, paid.replace("\n", " "))
    for (model, tier), d in acc.items():
        if "in" not in d or "out" not in d:
            continue
        (pin, rin), (pout, rout) = d["in"], d["out"]
        reasons = []
        if re.search(r'prompts?\s*[<>]', rin + rout):
            reasons.append("tiered by prompt length; base tier captured")
        if re.search(r'through|starting|until', rin + rout, re.I):
            reasons.append("date-dependent/promo price; read raw_row")
        recs.append(dict(
            provider=ctx["provider"], model=model, section=tier or "(n/a)",
            input_per_1m=per_1m(pin[0], pin[2]), output_per_1m=per_1m(pout[0], pout[2]),
            currency=pin[1], unit_detected=pin[2] or "1M",
            needs_review=bool(reasons), review_reason="; ".join(reasons),
            raw_row=f"IN: {rin} || OUT: {rout}"))
    return recs


LONG_IN = re.compile(r'^(输入|input)', re.I)
LONG_OUT = re.compile(r'^(输出|output)', re.I)
BILLING_HEADINGS = ("按量后付费", "预付费", "后付费", "pay-as-you-go")


def _long_unit(text):
    """'美元/百万tokens' -> ('USD','1M'); '元/千tokens' -> ('CNY','1K')."""
    cur = "USD" if re.search(r'美元|USD|\$', text) else ("CNY" if re.search(r'元|CNY|RMB|¥|￥', text) else "")
    unit = "1M" if re.search(r'百万|1M|million', text, re.I) else ("1K" if re.search(r'千|1K', text) else "")
    return cur, unit


def parse_long(content, ctx):
    """Long-format tables (Baidu Qianfan): one row per sub-item, rowspan'd model
    cells, bare-number price and a separate unit column, e.g.
      [model, version, service, '输入', '1.4', '美元/百万tokens']
      ['输出', '5.6', '美元/百万tokens']"""
    recs, section, model = [], "", ""
    acc = {}  # (section, model) -> {"in": (val, cur, unit, raw), "out": ...}
    order = []
    for kind, payload in blocks_for(content):
        if kind == "heading":
            if payload in BILLING_HEADINGS and section:
                section = section.split(" / ")[0] + " / " + payload
            else:
                section = payload
            continue
        for cells in payload:
            # find the sub-item label followed by a bare number
            for i in range(len(cells) - 1):
                lab = cells[i].strip()
                if not (LONG_IN.match(lab) or LONG_OUT.match(lab) or "缓存" in lab
                        or "cache" in lab.lower()):
                    continue
                num = re.fullmatch(r'[0-9][0-9,]*(?:\.[0-9]+)?', cells[i + 1].strip())
                if not num:
                    continue
                if i >= 1:  # row carries the model cells
                    model = cells[1] if i >= 2 and cells[1] else cells[0]
                if "缓存" in lab or "cache" in lab.lower():
                    break  # cached-input price: not captured
                cur, unit = _long_unit(" ".join(cells[i + 2:]))
                key = (section, model)
                if key not in acc:
                    acc[key] = {}
                    order.append(key)
                acc[key]["in" if LONG_IN.match(lab) else "out"] = (
                    float(num.group(0).replace(",", "")), cur, unit,
                    " | ".join(c.replace("\n", " ") for c in cells))
                break
    for key in order:
        d = acc[key]
        if "in" not in d or "out" not in d:
            continue
        (vin, cin, uin, rin), (vout, cout, uout, rout) = d["in"], d["out"]
        reasons = []
        if not (uin or uout):
            reasons.append("per-unit not stated; assumed 1M")
        if not (cin or cout):
            reasons.append("currency not stated")
        if cin and cout and cin != cout:
            reasons.append("currency mismatch")
        recs.append(dict(
            provider=ctx["provider"], model=key[1], section=key[0],
            input_per_1m=per_1m(vin, uin), output_per_1m=per_1m(vout, uout or uin),
            currency=cin or cout, unit_detected=uin or uout or "1M",
            needs_review=bool(reasons), review_reason="; ".join(reasons),
            raw_row=f"IN: {rin} || OUT: {rout}"))
    return recs


PARSERS = {"generic": parse_generic, "gemini": parse_gemini, "long": parse_long}


def stamp_rows(recs, ctx):
    for r in recs:
        r.update(source_url=ctx["url"], source_type=ctx["source_type"],
                 retrieved_at=ctx["retrieved_at"], snapshot_sha256=ctx["sha"])
    return recs


# --------------------------------------------------------------------------
# LIVE mode
# --------------------------------------------------------------------------
def fetch_live(src):
    """Return (content, ext, method). Tries HTTP, then browser if allowed."""
    url, mode = src["url"], src["render"]
    if mode in ("no", "auto"):
        try:
            r = http_get(url)
            return r.text, ("md" if url.endswith(".md") else "html"), "http"
        except RuntimeError as e:
            if mode == "no":
                raise
            print(f"  http failed ({e}); trying browser")
    shot = RAW / src["provider"] / f"{now_utc()}.png"
    shot.parent.mkdir(parents=True, exist_ok=True)
    return fetch_rendered(url, shot), "html", "playwright"


def run_live(args):
    summary = []
    for src in SOURCES:
        if args.provider and src["provider"] not in args.provider:
            continue
        print(f"[live] {src['provider']}: {src['url']}")
        try:
            content, ext, method = fetch_live(src)
            recs = PARSERS[src["parser"]](content, dict(src))
            # auto mode: static HTML had no tables -> retry in a browser
            if not recs and src["render"] == "auto" and method == "http":
                print("  0 rows from static HTML; retrying with browser")
                shot = RAW / src["provider"] / f"{now_utc()}.png"
                content, ext, method = fetch_rendered(src["url"], shot), "html", "playwright"
                recs = PARSERS[src["parser"]](content, dict(src))
        except Exception as e:
            print(f"  ERROR: {e}")
            summary.append((src["provider"], "ERROR", str(e)[:80]))
            continue
        path, sha = save_snapshot(src["provider"], src["url"], content, ext, method)
        ctx = dict(url=src["url"], source_type="official_live",
                   retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   sha=sha)
        recs = stamp_rows(recs, ctx)
        append_csv(LIVE_CSV, recs, FIELDS)
        flagged = sum(r["needs_review"] for r in recs)
        print(f"  saved {path.name}; {len(recs)} rows ({flagged} need review)")
        summary.append((src["provider"], len(recs), flagged))
        time.sleep(POLITE_DELAY)
    print("\nSUMMARY  provider / rows / flagged")
    for s in summary:
        print("  ", *s)
    print(f"\nParsed rows -> {LIVE_CSV}")
    print("Providers with 0 rows: open the saved raw file/screenshot and adjust the parser or URL.")


# --------------------------------------------------------------------------
# HISTORY mode (Wayback Machine)
# --------------------------------------------------------------------------
def cdx_snapshots(url, since, until, collapse="timestamp:6"):
    params = {"url": url, "output": "json", "filter": "statuscode:200",
              "collapse": collapse, "fl": "timestamp,original",
              "from": since.replace("-", ""), "to": until.replace("-", "")}
    r = http_get("https://web.archive.org/cdx/search/cdx", params=params, timeout=90)
    try:
        rows = r.json()
    except ValueError:
        return []
    return rows[1:] if rows else []


CAND_FIELDS = ["provider", "snapshot", "url", "context", "line"]
CAND_PRICE = re.compile(r'(US\$|\$|¥|￥)\s*[0-9]')
CAND_UNIT = re.compile(r'token|1K|1M|million|千|百万|characters|/\s*image|/\s*min', re.I)
RAW_PRICE = re.compile(r'\$\s?[0-9]+(?:\.[0-9]+)?\s*(?:/|per)\s*1\s?[KkMm]\s*tokens?')


def text_candidates(html):
    """Price-looking lines from a page, each with `context`: the nearest preceding
    non-price labels (model name / heading), so a bare '$5.00 / 1M input tokens'
    can be tied to its model. Handles units split onto the next line
    ('$0.0200' + '/ 1K tokens'), drops the page's duplicate (desktop/mobile)
    copies, and falls back to scanning raw HTML/JSON when the visible text has
    no prices (client-rendered pages)."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
    out, seen = [], set()
    for i, line in enumerate(lines):
        if not CAND_PRICE.search(line):
            continue
        text = line
        if not CAND_UNIT.search(line):
            # unit on its own short line right after the price ("$0.03", "/ 1K tokens")
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not (CAND_UNIT.search(nxt) and len(nxt) <= 30):
                continue
            text = f"{line} {nxt}"
        ctx = []
        for prev in reversed(lines[max(0, i - 8):i]):
            if CAND_PRICE.search(prev) or len(prev) > 60:
                continue
            if (CAND_UNIT.search(prev) and len(prev) <= 30) or re.match(r'^(/|per\b)', prev):
                continue  # a unit fragment ("/ 1K tokens"), not a label
            ctx.insert(0, prev)
            if len(ctx) == 3:
                break
        key = (" > ".join(ctx), text[:300])
        if key not in seen:
            seen.add(key)
            out.append(dict(context=key[0], line=key[1]))
    if not out:  # client-rendered: prices only inside embedded JSON
        for m in RAW_PRICE.finditer(html):
            pre = re.sub(r'<[^>]+>|\\[nt"]|["{}]', " ", html[max(0, m.start() - 120):m.start()])
            key = (" ".join(pre.split())[-80:], m.group(0))
            if key not in seen:
                seen.add(key)
                out.append(dict(context=key[0], line=key[1]))
    return out[:400]


def run_history(args):
    since = args.since if len(args.since) > 4 else args.since + "-01"
    until = args.until if len(args.until) > 4 else args.until + "-12"
    total = fails = 0
    try:  # one quick probe so an unreachable archive fails in seconds, not minutes
        SESSION.get("https://web.archive.org/cdx/search/cdx",
                    params={"url": "example.com", "limit": "1"}, timeout=30)
    except requests.RequestException as e:
        print(f"Wayback Machine unreachable from this network ({type(e).__name__}); "
              "run history mode from another network.")
        gap_report(args.provider or list(HISTORY), int(since[:4]), int(until[:4]))
        return
    for provider, urls in HISTORY.items():
        if args.provider and provider not in args.provider:
            continue
        for url in urls:
            try:
                snaps = cdx_snapshots(url, since, until)
                time.sleep(WAYBACK_DELAY)
            except Exception as e:
                print(f"[history] {provider} {url}: CDX error {e}")
                continue
            if args.limit:
                snaps = snaps[-args.limit:]
            print(f"[history] {provider} {url}: {len(snaps)} monthly snapshots")
            for ts, original in snaps:
                d = RAW / provider
                tag = hashlib.md5(url.encode()).hexdigest()[:6]
                target = d / f"wb_{ts}_{tag}.html"
                if target.exists():
                    continue
                snap_url = f"https://web.archive.org/web/{ts}id_/{original}"
                try:
                    r = http_get(snap_url, timeout=90, tries=4, backoff=15)
                    fails = 0
                except Exception as e:
                    fails += 1
                    print(f"    {ts}: fetch failed ({e})")
                    if fails >= 2:
                        # Wayback blocks an IP that keeps going while rate-limited;
                        # every further request extends the block. Stop now.
                        print("\nSTOPPED: web.archive.org is refusing this IP (rate limit "
                              "block). Wait ~1 hour without visiting archive.org, then "
                              "re-run the same command; saved snapshots are skipped.")
                        gap_report(args.provider or list(HISTORY), int(since[:4]), int(until[:4]))
                        return
                    time.sleep(60)
                    continue
                html = r.content.decode("utf-8", errors="replace")
                path, sha = save_snapshot(provider, snap_url, html, "html", "wayback",
                                          stamp=f"wb_{ts}_{tag}")
                ctx = dict(provider=provider, authoritative=False, url=snap_url,
                           source_type="wayback",
                           retrieved_at=datetime.strptime(ts[:14], "%Y%m%d%H%M%S")
                           .replace(tzinfo=timezone.utc).isoformat(timespec="seconds"),
                           sha=sha)
                recs = []
                try:
                    recs = stamp_rows(parse_generic(html, ctx), ctx)
                except Exception as e:
                    print(f"    {ts}: parse error {e}")
                append_csv(HIST_CSV, recs, FIELDS)
                cands = [dict(provider=provider, snapshot=ts, url=snap_url, **c)
                         for c in text_candidates(html)]
                append_csv(CAND_CSV, cands, CAND_FIELDS)
                total += 1
                print(f"    {ts}: {len(recs)} table rows, {len(cands)} candidate lines")
                time.sleep(WAYBACK_DELAY)
    print(f"\nDone. {total} new snapshots. See {HIST_CSV} and {CAND_CSV}.")
    gap_report(args.provider or list(HISTORY), int(since[:4]), int(until[:4]))
    print("Old pages rarely have clean tables: use history_candidates.csv + raw HTML "
          "to hand-verify each historical price.")


def run_reparse(args):
    """Rebuild prices_history.csv and history_candidates.csv from the Wayback
    pages already saved in raw/ - no network. Use after a parser change."""
    urls = {}
    if MANIFEST.exists():
        for l in MANIFEST.open(encoding="utf-8"):
            m = json.loads(l)
            urls[Path(m["file"]).name] = (m["url"], m["sha256"])
    files = sorted(RAW.glob("*/wb_*.html"))
    for f in (HIST_CSV, CAND_CSV):
        if f.exists():
            f.replace(f.with_suffix(".csv.bak"))
    n_rows = n_cands = 0
    for f in files:
        provider, ts = f.parent.name, f.name.split("_")[1]
        if args.provider and provider not in args.provider:
            continue
        html = f.read_text(encoding="utf-8", errors="replace")
        url, sha = urls.get(f.name, ("", hashlib.sha256(html.encode()).hexdigest()))
        ctx = dict(provider=provider, authoritative=False, url=url, source_type="wayback",
                   retrieved_at=datetime.strptime(ts[:14], "%Y%m%d%H%M%S")
                   .replace(tzinfo=timezone.utc).isoformat(timespec="seconds"), sha=sha)
        try:
            recs = stamp_rows(parse_generic(html, ctx), ctx)
        except Exception as e:
            print(f"  {f.name}: parse error {e}")
            recs = []
        cands = [dict(provider=provider, snapshot=ts, url=url, **c)
                 for c in text_candidates(html)]
        append_csv(HIST_CSV, recs, FIELDS)
        append_csv(CAND_CSV, cands, CAND_FIELDS)
        n_rows, n_cands = n_rows + len(recs), n_cands + len(cands)
        print(f"  {provider} {ts}: {len(recs)} table rows, {len(cands)} candidate lines")
    print(f"\nReparsed {len(files)} saved snapshots: {n_rows} table rows, "
          f"{n_cands} candidate lines (old CSVs kept as *.csv.bak).")
    years = [int(f.name[3:7]) for f in files] or [datetime.now().year]
    gap_report(args.provider or list(HISTORY), min(years), max(years))


def gap_report(providers, y0, y1):
    """Provider x year table of archived snapshots on disk (raw/<p>/wb_YYYY*)
    and parsed table rows; 0/0 cells are the gaps to fill by hand."""
    rows = {}
    if HIST_CSV.exists():
        for r in csv.DictReader(HIST_CSV.open(encoding="utf-8")):
            k = (r["provider"], r["retrieved_at"][:4])
            rows[k] = rows.get(k, 0) + 1
    years = list(range(y0, y1 + 1))
    print("\nCOVERAGE  snapshots/parsed-rows per year (0/0 = gap)")
    print("  " + "provider".ljust(10) + "".join(str(y).rjust(10) for y in years))
    for p in providers:
        snaps = [f.name[3:7] for f in (RAW / p).glob("wb_*.html")] if (RAW / p).exists() else []
        cells = [f"{snaps.count(str(y))}/{rows.get((p, str(y)), 0)}" for y in years]
        print("  " + p.ljust(10) + "".join(c.rjust(10) for c in cells))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("live", help="snapshot current official pricing pages")
    a.add_argument("--provider", nargs="*", help="e.g. openai anthropic")
    b = sub.add_parser("history", help="pull archived pages from the Wayback Machine")
    b.add_argument("--provider", nargs="*")
    b.add_argument("--since", default="2020-06", help="YYYY or YYYY-MM")
    b.add_argument("--until", default=datetime.now().strftime("%Y-%m"))
    b.add_argument("--limit", type=int, help="only the most recent N snapshots per URL")
    c = sub.add_parser("reparse", help="rebuild history CSVs from saved raw pages (offline)")
    c.add_argument("--provider", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"live": run_live, "history": run_history, "reparse": run_reparse}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
