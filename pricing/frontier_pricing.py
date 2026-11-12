#!/usr/bin/env python3
"""Historical API pricing for frontier AI labs (US, Chinese, European) and open-weight models.

Prices are USD per 1M tokens (input / output) at the date shown. Each row in
data/llm_api_prices.csv is one *price event*: a launch, a price cut or a change.

"Blended" price = (3 x input + 1 x output) / 4, the common 3:1 input:output mix.

Standard library only (Python 3.9+). Examples:

    python3 pricing/frontier_pricing.py table                      # everything, by date
    python3 pricing/frontier_pricing.py table --country CN         # Chinese labs only
    python3 pricing/frontier_pricing.py table --open               # open-weight models only
    python3 pricing/frontier_pricing.py table --lab openai --lab anthropic --since 2024
    python3 pricing/frontier_pricing.py timeline --lab anthropic   # per-lab price history
    python3 pricing/frontier_pricing.py summary                    # one line per lab
    python3 pricing/frontier_pricing.py yearly                     # min / median / max per year
    python3 pricing/frontier_pricing.py cheapest --top 15          # cheapest models ever launched
    python3 pricing/frontier_pricing.py chart --out prices.svg     # log-scale chart (SVG)
    python3 pricing/frontier_pricing.py export --format json --out prices.json
    python3 pricing/frontier_pricing.py live --save                # today's prices from OpenRouter
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "data" / "llm_api_prices.csv"
SNAPSHOT_DIR = HERE / "data" / "snapshots"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

# OpenRouter id prefix -> lab name used in the dataset
OPENROUTER_LABS = {
    "openai": ("OpenAI", "US"),
    "anthropic": ("Anthropic", "US"),
    "google": ("Google", "US"),
    "x-ai": ("xAI", "US"),
    "meta-llama": ("Meta", "US"),
    "amazon": ("Amazon", "US"),
    "mistralai": ("Mistral", "FR"),
    "cohere": ("Cohere", "CA"),
    "deepseek": ("DeepSeek", "CN"),
    "qwen": ("Alibaba", "CN"),
    "bytedance": ("ByteDance", "CN"),
    "bytedance-seed": ("ByteDance", "CN"),
    "baidu": ("Baidu", "CN"),
    "moonshotai": ("Moonshot AI", "CN"),
    "z-ai": ("Zhipu AI (Z.ai)", "CN"),
    "thudm": ("Zhipu AI (Z.ai)", "CN"),
    "minimax": ("MiniMax", "CN"),
    "tencent": ("Tencent", "CN"),
}

LAB_COLORS = {
    "OpenAI": "#10a37f", "Anthropic": "#d97757", "Google": "#4285f4", "xAI": "#555555",
    "Meta": "#0866ff", "Amazon": "#ff9900", "Mistral": "#fa520f", "Cohere": "#39594d",
    "DeepSeek": "#00a3a3", "Alibaba": "#7b3fe4", "ByteDance": "#e6194b", "Baidu": "#2932e1",
    "Moonshot AI": "#111111", "Zhipu AI (Z.ai)": "#3cb44b", "MiniMax": "#f032e6",
}


# --------------------------------------------------------------------------- data

def load(path: Path = DATA_FILE) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["input_per_m"] = float(r["input_per_m"])
            r["output_per_m"] = float(r["output_per_m"])
            r["blended"] = round((3 * r["input_per_m"] + r["output_per_m"]) / 4, 4)
            r["open_weights"] = r["open_weights"].strip().lower() == "yes"
            r["date"] = dt.date.fromisoformat(r["price_date"]) if r["price_date"] else None
            rows.append(r)
    return rows


def sort_key(r: dict):
    # undated rows (known price, release date not yet filled in) go last
    return (r["date"] is None, r["date"] or dt.date.max, r["lab"], r["model"])


def apply_filters(rows: list[dict], a) -> list[dict]:
    out = rows
    if getattr(a, "lab", None):
        wanted = [x.lower() for x in a.lab]
        out = [r for r in out if any(w in r["lab"].lower() for w in wanted)]
    if getattr(a, "country", None):
        cs = {c.upper() for c in a.country}
        out = [r for r in out if r["country"].upper() in cs]
    if getattr(a, "open", False):
        out = [r for r in out if r["open_weights"]]
    if getattr(a, "closed", False):
        out = [r for r in out if not r["open_weights"]]
    if getattr(a, "since", None):
        since = dt.date(int(a.since), 1, 1) if len(a.since) == 4 else dt.date.fromisoformat(a.since)
        out = [r for r in out if r["date"] is None or r["date"] >= since]
    if getattr(a, "min_confidence", None):
        rank = {"low": 0, "medium": 1, "high": 2}
        out = [r for r in out if rank.get(r["confidence"], 0) >= rank[a.min_confidence]]
    if getattr(a, "launches_only", False):
        out = [r for r in out if r["event"] == "launch"]
    return sorted(out, key=sort_key)


# --------------------------------------------------------------------------- output helpers

def money(x: float) -> str:
    if x >= 10:
        return f"${x:,.2f}".rstrip("0").rstrip(".")
    if x >= 0.1:
        return f"${x:.2f}"
    return f"${x:.3f}"


def print_table(headers: list[str], rows: list[list], aligns: str | None = None) -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]
    aligns = aligns or "l" * len(headers)

    def fmt(cells):
        return "  ".join(str(c).rjust(w) if al == "r" else str(c).ljust(w)
                         for c, w, al in zip(cells, widths, aligns))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def row_cells(r: dict) -> list:
    return [
        r["price_date"] or "(date?)", r["lab"], r["country"], r["model"], r["event"],
        money(r["input_per_m"]), money(r["output_per_m"]), money(r["blended"]),
        "open" if r["open_weights"] else "", r["confidence"],
    ]


HEADERS = ["date", "lab", "cc", "model", "event", "in/1M", "out/1M", "blend/1M", "weights", "conf"]
ALIGNS = "lllllrrrll"


# --------------------------------------------------------------------------- commands

def cmd_table(a) -> None:
    rows = apply_filters(load(), a)
    if a.sort == "price":
        rows.sort(key=lambda r: r["blended"])
    print_table(HEADERS, [row_cells(r) for r in rows], ALIGNS)
    print(f"\n{len(rows)} price events. Prices: USD per 1M tokens; blend = (3*in + out)/4.")


def cmd_timeline(a) -> None:
    rows = apply_filters(load(), a)
    by_lab = defaultdict(list)
    for r in rows:
        by_lab[r["lab"]].append(r)
    for lab, items in by_lab.items():
        print(f"\n=== {lab} ({items[0]['country']}) ===")
        first = next((r["blended"] for r in items if r["date"]), items[0]["blended"])
        cells = []
        for r in items:
            ratio = r["blended"] / first
            vs = f"{ratio:.2f}x" if ratio >= 0.1 else f"1/{1 / ratio:.0f}"
            cells.append([r["price_date"] or "(date?)", r["model"], r["event"],
                          money(r["input_per_m"]), money(r["output_per_m"]),
                          money(r["blended"]), vs, r["notes"]])
        print_table(["date", "model", "event", "in/1M", "out/1M", "blend", "vs first", "notes"],
                    cells, "lllrrrrl")


def cmd_summary(a) -> None:
    rows = apply_filters(load(), a)
    by_lab = defaultdict(list)
    for r in rows:
        by_lab[r["lab"]].append(r)
    cells = []
    for lab, items in sorted(by_lab.items(), key=lambda kv: min((r["date"] or dt.date.max) for r in kv[1])):
        dated = [r for r in items if r["date"]]
        hi = max(items, key=lambda r: r["blended"])
        lo = min(items, key=lambda r: r["blended"])
        latest = max(dated, key=lambda r: r["date"]) if dated else items[-1]
        cells.append([
            lab, items[0]["country"], len(items),
            min(r["price_date"] for r in dated) if dated else "?",
            f"{hi['model']} {money(hi['blended'])}",
            f"{lo['model']} {money(lo['blended'])}",
            sum(r["open_weights"] for r in items),
            f"{latest['model']} ({latest['price_date']})",
        ])
    print_table(["lab", "cc", "events", "first API price", "priciest (blend)", "cheapest (blend)",
                 "open", "latest dated"], cells, "llrlllrl")


def cmd_yearly(a) -> None:
    rows = [r for r in apply_filters(load(), a) if r["date"]]
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["date"].year].append(r)
    cells = []
    for y in sorted(by_year):
        items = by_year[y]
        b = [r["blended"] for r in items]
        lo = min(items, key=lambda r: r["blended"])
        hi = max(items, key=lambda r: r["blended"])
        cn = [r["blended"] for r in items if r["country"] == "CN"]
        us = [r["blended"] for r in items if r["country"] == "US"]
        cells.append([y, len(items), money(min(b)), money(statistics.median(b)), money(max(b)),
                      money(statistics.median(us)) if us else "-",
                      money(statistics.median(cn)) if cn else "-",
                      f"{lo['model']} ({lo['lab']})", f"{hi['model']} ({hi['lab']})"])
    print_table(["year", "n", "min", "median", "max", "US median", "CN median", "cheapest", "priciest"],
                cells, "rrrrrrrll")
    print("\nAll values are blended USD per 1M tokens across price events in that year.")


def cmd_cheapest(a) -> None:
    rows = apply_filters(load(), a)
    rows.sort(key=lambda r: r["blended"], reverse=a.expensive)
    print_table(HEADERS, [row_cells(r) for r in rows[: a.top]], ALIGNS)


def cmd_export(a) -> None:
    rows = apply_filters(load(), a)
    out = open(a.out, "w", newline="", encoding="utf-8") if a.out else sys.stdout
    fields = ["price_date", "release_date", "lab", "country", "model", "event", "input_per_m",
              "output_per_m", "blended", "open_weights", "confidence", "notes"]
    try:
        if a.format == "json":
            json.dump([{k: r[k] for k in fields} for r in rows], out, indent=2)
            out.write("\n")
        elif a.format == "csv":
            w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        else:  # markdown
            out.write("| " + " | ".join(HEADERS) + " |\n")
            out.write("|" + "|".join("---:" if al == "r" else "---" for al in ALIGNS) + "|\n")
            for r in rows:
                out.write("| " + " | ".join(str(c) for c in row_cells(r)) + " |\n")
    finally:
        if a.out:
            out.close()
            print(f"Wrote {len(rows)} rows to {a.out}", file=sys.stderr)


def cmd_chart(a) -> None:
    """Log-scale scatter of blended price over time, one colour per lab. Pure SVG, no deps."""
    import math

    rows = [r for r in apply_filters(load(), a) if r["date"]]
    if not rows:
        sys.exit("No dated rows match the filters.")
    W, H, L, R, T, B = 1200, 700, 80, 220, 50, 60
    d0 = min(r["date"] for r in rows).toordinal() - 30
    d1 = max(r["date"] for r in rows).toordinal() + 30
    lo = math.floor(math.log10(min(r["blended"] for r in rows)))
    hi = math.ceil(math.log10(max(r["blended"] for r in rows)))

    def x(d):
        return L + (d.toordinal() - d0) / (d1 - d0) * (W - L - R)

    def y(p):
        return T + (hi - math.log10(p)) / (hi - lo) * (H - T - B)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="28" font-size="18" font-weight="600">LLM API price history (blended USD / 1M tokens, log scale)</text>']
    for e in range(lo, hi + 1):
        yy = y(10 ** e)
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#ddd"/>')
        s.append(f'<text x="{L - 8}" y="{yy + 4:.1f}" text-anchor="end">{money(10 ** e)}</text>')
    for yr in range(dt.date.fromordinal(d0).year + 1, dt.date.fromordinal(d1).year + 1):
        xx = x(dt.date(yr, 1, 1))
        s.append(f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" y2="{H - B}" stroke="#eee"/>')
        s.append(f'<text x="{xx:.1f}" y="{H - B + 18}" text-anchor="middle">{yr}</text>')
    labs = sorted({r["lab"] for r in rows}, key=lambda l: min(r["date"] for r in rows if r["lab"] == l))
    for r in rows:
        c = LAB_COLORS.get(r["lab"], "#888")
        fill = "white" if r["open_weights"] else c
        tip = f"{r['price_date']} {r['lab']} {r['model']}: in {money(r['input_per_m'])} / out {money(r['output_per_m'])}"
        s.append(f'<circle cx="{x(r["date"]):.1f}" cy="{y(r["blended"]):.1f}" r="5" fill="{fill}" '
                 f'stroke="{c}" stroke-width="2" opacity="0.9"><title>{tip}</title></circle>')
        if a.labels:
            s.append(f'<text x="{x(r["date"]) + 7:.1f}" y="{y(r["blended"]) + 4:.1f}" font-size="9" fill="#444">{r["model"]}</text>')
    for i, lab in enumerate(labs):
        yy = T + 10 + i * 20
        c = LAB_COLORS.get(lab, "#888")
        s.append(f'<circle cx="{W - R + 20}" cy="{yy}" r="5" fill="{c}"/>')
        s.append(f'<text x="{W - R + 32}" y="{yy + 4}">{lab}</text>')
    yy = T + 20 + len(labs) * 20
    s.append(f'<circle cx="{W - R + 20}" cy="{yy}" r="5" fill="white" stroke="#333" stroke-width="2"/>')
    s.append(f'<text x="{W - R + 32}" y="{yy + 4}">hollow = open weights</text>')
    s.append("</svg>")
    Path(a.out).write_text("\n".join(s), encoding="utf-8")
    print(f"Wrote {a.out} ({len(rows)} points). Hover a point in a browser for details.")


def fetch_openrouter(timeout: int = 30) -> list[dict]:
    req = urllib.request.Request(OPENROUTER_URL, headers={"User-Agent": "frontier-pricing-script"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    today = dt.date.today().isoformat()
    out = []
    for m in payload.get("data", []):
        mid = m.get("id", "")
        prefix = mid.split("/", 1)[0]
        if prefix not in OPENROUTER_LABS or mid.endswith(":free"):
            continue
        p = m.get("pricing") or {}
        try:
            inp = float(p.get("prompt", 0)) * 1_000_000
            outp = float(p.get("completion", 0)) * 1_000_000
        except (TypeError, ValueError):
            continue
        if inp <= 0 and outp <= 0:
            continue
        lab, cc = OPENROUTER_LABS[prefix]
        created = m.get("created")
        out.append({
            "snapshot_date": today,
            "lab": lab, "country": cc, "openrouter_id": mid, "name": m.get("name", mid),
            "listed_date": dt.datetime.fromtimestamp(created, dt.timezone.utc).date().isoformat() if created else "",
            "input_per_m": round(inp, 4), "output_per_m": round(outp, 4),
            "blended": round((3 * inp + outp) / 4, 4),
            "context_length": m.get("context_length", ""),
        })
    return sorted(out, key=lambda r: (r["lab"], r["listed_date"]))


def cmd_live(a) -> None:
    try:
        rows = fetch_openrouter()
    except Exception as e:  # network blocked, proxy, etc.
        sys.exit(f"Could not reach {OPENROUTER_URL}: {e}")
    if a.lab:
        rows = [r for r in rows if any(w.lower() in r["lab"].lower() for w in a.lab)]
    if a.country:
        rows = [r for r in rows if r["country"] in {c.upper() for c in a.country}]
    print_table(["lab", "cc", "openrouter id", "listed", "in/1M", "out/1M", "blend"],
                [[r["lab"], r["country"], r["openrouter_id"], r["listed_date"], money(r["input_per_m"]),
                  money(r["output_per_m"]), money(r["blended"])] for r in rows], "llllrrr")
    print(f"\n{len(rows)} models. Note: OpenRouter prices for open-weight models are the cheapest "
          "routed provider, not the lab's own API.")

    previous = sorted(SNAPSHOT_DIR.glob("openrouter_*.csv")) if SNAPSHOT_DIR.exists() else []
    if previous:
        with previous[-1].open(newline="", encoding="utf-8") as f:
            old = {r["openrouter_id"]: r for r in csv.DictReader(f)}
        changes = []
        for r in rows:
            o = old.get(r["openrouter_id"])
            if o and (abs(float(o["input_per_m"]) - r["input_per_m"]) > 1e-9
                      or abs(float(o["output_per_m"]) - r["output_per_m"]) > 1e-9):
                changes.append([r["openrouter_id"], f"{money(float(o['input_per_m']))}/{money(float(o['output_per_m']))}",
                                f"{money(r['input_per_m'])}/{money(r['output_per_m'])}"])
        new = [r["openrouter_id"] for r in rows if r["openrouter_id"] not in old]
        print(f"\nCompared with {previous[-1].name}: {len(changes)} price changes, {len(new)} new models.")
        if changes:
            print_table(["model", "was in/out", "now in/out"], changes)
        if new:
            print("New: " + ", ".join(new))

    if a.save:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path = SNAPSHOT_DIR / f"openrouter_{dt.date.today().isoformat()}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["snapshot_date"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved snapshot to {path}. Run this regularly (e.g. weekly cron) to build "
              "your own price history going forward.")


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def filters(sp):
        sp.add_argument("--lab", action="append", help="lab name substring; repeatable (e.g. --lab openai --lab deepseek)")
        sp.add_argument("--country", action="append", help="country code: US, CN, FR, CA; repeatable")
        g = sp.add_mutually_exclusive_group()
        g.add_argument("--open", action="store_true", help="open-weight models only")
        g.add_argument("--closed", action="store_true", help="closed models only")
        sp.add_argument("--since", help="YYYY or YYYY-MM-DD")
        sp.add_argument("--min-confidence", choices=["low", "medium", "high"])
        sp.add_argument("--launches-only", action="store_true", help="skip price-cut/change events")

    sp = sub.add_parser("table", help="all price events"); filters(sp)
    sp.add_argument("--sort", choices=["date", "price"], default="date"); sp.set_defaults(fn=cmd_table)
    sp = sub.add_parser("timeline", help="per-lab price history"); filters(sp); sp.set_defaults(fn=cmd_timeline)
    sp = sub.add_parser("summary", help="one line per lab"); filters(sp); sp.set_defaults(fn=cmd_summary)
    sp = sub.add_parser("yearly", help="price distribution per year"); filters(sp); sp.set_defaults(fn=cmd_yearly)
    sp = sub.add_parser("cheapest", help="cheapest (or --expensive) models"); filters(sp)
    sp.add_argument("--top", type=int, default=20); sp.add_argument("--expensive", action="store_true")
    sp.set_defaults(fn=cmd_cheapest)
    sp = sub.add_parser("export", help="export to csv/json/md"); filters(sp)
    sp.add_argument("--format", choices=["csv", "json", "md"], default="csv"); sp.add_argument("--out")
    sp.set_defaults(fn=cmd_export)
    sp = sub.add_parser("chart", help="write an SVG chart"); filters(sp)
    sp.add_argument("--out", default="llm_price_history.svg"); sp.add_argument("--labels", action="store_true")
    sp.set_defaults(fn=cmd_chart)
    sp = sub.add_parser("live", help="current prices from OpenRouter (needs internet)")
    sp.add_argument("--lab", action="append"); sp.add_argument("--country", action="append")
    sp.add_argument("--save", action="store_true", help="save snapshot to data/snapshots/")
    sp.set_defaults(fn=cmd_live)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
