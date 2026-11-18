"""Part A adapter: turn the raw AI API price panel into a clean model x month price panel.

Inputs
------
* AI_API_price_panel.xlsx, sheet "Panel (USD)" (3 sources: W = Wayback archived official
  page, L = LiteLLM git history, Live = live scrape), CNY already converted to USD.
* Optionally the raw live-scrape CSV (prices.csv: provider, model, section, input_per_1m, ...),
  added as extra Live rows.

What the cleaner does (each step is logged in the cleaning report)
--------------------------------------------------------------------
1. Classify every row: maker (who built the model) vs host (who sells it), modality
   (text / vision / audio / image / embedding / ...), price tier (standard / batch / flex /
   priority / fine-tune / unknown), market (INTL vs CN), region rank, thinking mode,
   long-context tier, zero / per-character / promo prices.
2. Keep standard, base-context, non-zero, token-priced text-model rows. When a model key
   has rows whose published name *is* the key (e.g. "gpt-4o"), dated variants
   ("gpt-4o-2024-05-13") from the same source are dropped.
3. Archived OpenAI 2026 pages list Standard/Batch/Flex/Priority without labels. Each
   unlabelled group is resolved to the candidate closest to a labelled standard price
   for the same model within 200 days (another source or date); otherwise dropped.
4. One observation per model x market x date x source (best region, non-thinking first,
   lower-median price if several remain), then per date W/Live beat L.
5. LiteLLM rows only enter as first_seen/changed events (cache_changed only updates the
   cached-input price) so LiteLLM's lagged file does not undo a cut seen on an archived page.
6. Monthly panel: price in force at month end, carried forward at most `max_carry_months`
   after the last observation and never past a (non-suspect) LiteLLM removal.
7. Chained matched-model Jevons price index (overall, by maker, by market).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------- vocab
MAKER_RULES = [
    (r"^(ft:)?(gpt|o1|o3|o4|chatgpt|codex|text-|davinci|babbage|curie|ada|omni-moderation|"
     r"computer-use|daybreak|\d+k-context|chat-latest|gpt-35)", "openai"),
    (r"^claude", "anthropic"),
    (r"^(gemini|gemma|learnlm|learnim|lyria|nano-banana|deep-research|contributor)", "google"),
    (r"^(qwen|qwq|通义|tongyi)", "alibaba"),
    (r"^deepseek", "deepseek"),
    (r"^(glm|chatglm)", "zhipu"),
    (r"^kimi", "moonshot"),
    (r"^minimax", "minimax"),
    (r"^(ernie|eb-|qianfan|tao-|embedding-v1|推理服务)", "baidu"),
    (r"^(llama|meta-llama|codellama)", "meta"),
    (r"^(mixtral|mistral)", "mistral"),
    (r"^yi-", "01ai"),
    (r"^(bloomz|aquilachat|sqlcoder|xuanyuan|chatlaw|bge|bce)", "other_open"),
]
MAKER_COUNTRY = {"openai": "US", "anthropic": "US", "google": "US", "meta": "US", "mistral": "FR",
                 "alibaba": "CN", "deepseek": "CN", "zhipu": "CN", "moonshot": "CN", "minimax": "CN",
                 "baidu": "CN", "01ai": "CN"}

MODALITY_RULES = [  # (regex on key + published + section, modality) - first match wins
    (r"^ft:|fine-tun", "fine_tune"),
    (r"embed|rerank|bge-|bce-|向量", "embedding"),
    (r"moderation", "moderation"),
    (r"image|banana|dall-e|imagen|images", "image"),
    (r"tts|transcri|audio|speech|realtime|live-translate|livetranslate|lyria|-live|live-|whisper", "audio"),
    (r"\bvl\b|-vl-|-vl$|vision|ocr|omni", "vision_multimodal"),
    (r"robotics|computer-use|deep-research|search|intent|context$|^contributor$|推理服务|decisions",
     "other_tool"),
]


def maker_of(key: str, host: str) -> str:
    for pat, mk in MAKER_RULES:
        if re.search(pat, key):
            return mk
    return host


def model_key(published: str, strip_date: bool = True) -> str:
    """Normalise a published model name to the Part A 'Model key' convention."""
    s = str(published).strip().lower()
    s = s.split("/")[-1]                              # dashscope/qwen-plus -> qwen-plus
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s)       # [thinking] (256K<Token≤1M)
    s = s.strip().replace("_", "-").replace(".", "-")
    s = re.sub(r"\s+", "-", s)
    if strip_date:
        s = re.sub(r"-(20\d{2}-\d{2}-\d{2}|20\d{6})$", "", s)  # dated snapshots
    s = re.sub(r"-+", "-", s).strip("-")
    m = re.match(r"^claude-(\d+(?:-\d+)?)-(opus|sonnet|haiku)(.*)$", s)
    if m:
        s = f"claude-{m.group(2)}-{m.group(1)}{m.group(3)}"
    return s


def _modality(text: str) -> str:
    for pat, mod in MODALITY_RULES:
        if re.search(pat, text):
            return mod
    return "text"


def _tier(section: str, flags: str, l_section: bool) -> str:
    s = f"{section} {flags}".lower()
    if "tier not labelled" in s or "tier unknown" in s:
        return "unknown"
    if "fine-tun" in s:
        return "fine_tune"
    if re.search(r"batch|批量", s):
        return "batch"
    if "flex" in s:
        return "flex"
    if "priority" in s:
        return "priority"
    if re.search(r"/\s*fast\b|^fast\b|fast mode", s):
        return "priority"
    if l_section and section.strip().lower() not in ("chat", "responses", "completion"):
        return "non_text_endpoint"
    return "standard"


def _market(section: str, currency: str, source: str, host: str) -> tuple[str, int]:
    s = str(section).lower()
    if currency == "CNY" or "china (beijing)" in s or "按量" in s or re.search(r"[一-鿿]", s):
        if "international" not in s and "singapore" not in s:
            return "CN", 0
    if "international" in s or "singapore" in s:
        return "INTL", 0
    if "united states" in s and s.rstrip().endswith("united states"):
        return "INTL", 1
    if "global" in s or "frankfurt" in s or "tokyo" in s:
        return "INTL", 2
    if "hong kong" in s:
        return "INTL", 3
    return "INTL", 0


def _higher_context(published: str, flags: str) -> bool:
    if "higher input-length tier" in flags.lower():
        return True
    m = re.search(r"\(\s*(\d+)\s*k\s*<\s*token", published.lower())
    return bool(m and int(m.group(1)) > 0) or bool(re.search(r"\(\s*>\s*\d+k", published.lower()))


def _thinking(published: str) -> float:
    p = published.lower()
    if "[non-thinking]" in p:
        return 0.0
    if "[thinking]" in p:
        return 1.0
    return np.nan


# ---------------------------------------------------------------------------- load
XLSX_COLS = {
    "Source": "source", "Provider": "host", "Model (as published)": "published", "Model key": "key",
    "Date": "date", "Section / tier": "section", "Event": "event", "Input USD / 1M": "price_in",
    "Output USD / 1M": "price_out", "Cached input USD / 1M": "price_cached_in",
    "Original currency": "currency", "Unit on page": "unit", "Flags": "flags",
}


def load_xlsx(path, sheet: str = "Panel (USD)") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    missing = set(XLSX_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} [{sheet}] is missing columns {missing}")
    df = df.rename(columns=XLSX_COLS)[list(XLSX_COLS.values())]
    sheets = pd.ExcelFile(path).sheet_names
    if "Panel (original)" in sheets:
        orig = pd.read_excel(path, sheet_name="Panel (original)")
        if len(orig) != len(df) or not (orig["Model key"].astype(str).values == df["key"].astype(str).values).all():
            raise ValueError("'Panel (original)' and 'Panel (USD)' rows are not aligned")
        df["price_in_local"] = orig["Input / 1M tokens"].values
        df["price_out_local"] = orig["Output / 1M tokens"].values
        df["source_url"] = orig["Source URL"].values
    else:
        df["price_in_local"], df["price_out_local"], df["source_url"] = df["price_in"], df["price_out"], ""
    df["origin_file"] = Path(path).name
    return df


def load_live_csv(path, key_map: dict[str, str] | None = None) -> pd.DataFrame:
    c = pd.read_csv(path)
    key_map = key_map or {}
    out = pd.DataFrame({
        "source": "Live", "host": c["provider"], "published": c["model"],
        "key": [key_map.get(str(m), model_key(m)) for m in c["model"]],
        "date": pd.to_datetime(c["retrieved_at"], utc=True).dt.tz_localize(None).dt.normalize(),
        "section": c["section"], "event": "live",
        "price_in": c["input_per_1m"], "price_out": c["output_per_1m"], "price_cached_in": np.nan,
        "price_in_local": c["input_per_1m"], "price_out_local": c["output_per_1m"],
        "currency": c["currency"], "unit": "per " + c["unit_detected"].astype(str),
        "flags": c["review_reason"].fillna(""), "source_url": c["source_url"],
        "origin_file": Path(path).name,
    })
    nonusd = out["currency"] != "USD"
    if nonusd.any():
        raise ValueError(f"{path}: {nonusd.sum()} non-USD rows; convert before loading")
    return out


def load(xlsx=None, live_csv=None) -> pd.DataFrame:
    parts = []
    if xlsx:
        parts.append(load_xlsx(xlsx))
    if live_csv:
        km = dict(zip(parts[0]["published"].astype(str), parts[0]["key"])) if parts else {}
        parts.append(load_live_csv(live_csv, km))
    if not parts:
        raise ValueError("Give at least one Part A file")
    raw = pd.concat(parts, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    for c in ["host", "published", "key", "section", "event", "currency", "unit", "flags", "source"]:
        raw[c] = raw[c].fillna("").astype(str)
    raw["key"] = raw["key"].str.strip().str.lower()
    # identical row from xlsx Live and csv Live -> keep one
    dedup = ["source", "host", "published", "date", "section", "price_in", "price_out"]
    raw = raw.drop_duplicates(dedup).reset_index(drop=True)
    return raw


# ---------------------------------------------------------------------------- classify
LITELLM_RESELLER_PREFIXES = {"azure": "azure", "openrouter": "openrouter", "bedrock": "bedrock",
                             "vertex_ai": "vertex"}


def classify(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    # LiteLLM namespaces: "fast/" = priority tier, "us/" = US-only inference premium,
    # "azure/" "openrouter/" = resellers (another host), "gemini/" "dashscope/" = provider namespace
    parts = d["published"].str.lower().str.split("/").map(lambda x: x[:-1] if len(x) > 1 else [])
    for pre, host in LITELLM_RESELLER_PREFIXES.items():
        d.loc[parts.map(lambda x, p=pre: p in x), "host"] = host
    d["maker"] = [maker_of(k, h) for k, h in zip(d["key"], d["host"])]
    d["third_party_host"] = d["maker"] != d["host"]
    text = (d["key"] + " " + d["published"].str.lower() + " " + d["section"].str.lower())
    d["modality"] = text.map(_modality)
    d["tier"] = [_tier(s, f, src == "L") for s, f, src in zip(d["section"], d["flags"], d["source"])]
    d.loc[parts.map(lambda x: "fast" in x), "tier"] = "priority"
    mk = [_market(s, c, src, h) for s, c, src, h in zip(d["section"], d["currency"], d["source"], d["host"])]
    d["market"] = [m for m, _ in mk]
    d["region_rank"] = [r for _, r in mk]
    d.loc[parts.map(lambda x: "us" in x or "eu" in x), "region_rank"] = 1
    d["higher_context"] = [_higher_context(p, f) for p, f in zip(d["published"], d["flags"])]
    d["thinking"] = d["published"].map(_thinking)
    # exact alias = published name is the key itself, not a dated snapshot of it
    d["exact_alias"] = d["published"].map(lambda x: model_key(x, strip_date=False)) == d["key"]
    d["is_zero"] = (d["price_in"].fillna(0) == 0) & (d["price_out"].fillna(0) == 0)
    d["per_character"] = d["unit"].str.contains("CHARACTER", case=False)
    d["is_promo"] = d["flags"].str.contains("promo|date-dependent", case=False)
    d["is_suspect"] = d["flags"].str.contains("SUSPECT", case=False)

    # keys that are tier/context labels or page fragments rather than models
    d["not_a_model"] = d["key"].str.contains(r"token|^\d+k-context$|^\d+k[<≤]|^contributor$|^推理服务$|系列$",
                                             regex=True)
    reason = pd.Series("", index=d.index)

    def mark(mask, why):
        nonlocal reason
        reason = reason.where(reason != "", np.where(mask, why, ""))

    mark(d["not_a_model"], "not_a_model_key")
    mark(d["event"] == "removed", "removal_event")
    mark(d["modality"] != "text", "non_text_modality")
    mark(~d["tier"].isin(["standard", "unknown"]), "non_standard_tier")
    mark(d["higher_context"], "higher_context_tier")
    mark(d["per_character"], "priced_per_character")
    mark(d["is_zero"], "zero_price")
    mark(d["price_in"].isna() | d["price_out"].isna(), "missing_price")
    d["drop_reason"] = reason
    return d


# ---------------------------------------------------------------------------- select
def _pick_row(g: pd.DataFrame) -> pd.Series:
    """Choose one row from candidates of one key x market x date x source."""
    g = g.assign(_think=g["thinking"].fillna(0.5))
    g = g.sort_values(["region_rank", "_think"])
    best = g[(g["region_rank"] == g["region_rank"].iloc[0]) & (g["_think"] == g["_think"].iloc[0])]
    best = best.sort_values("price_in")
    return best.iloc[(len(best) - 1) // 2]  # lower median


def _resolve_unknown(cands: pd.DataFrame, ref: pd.DataFrame, max_days: int = 400,
                     tol: float = np.log(1.25)) -> pd.Series | None:
    """Pick the unlabelled-tier candidate closest to a labelled standard price."""
    if ref.empty:
        return None
    date = pd.Timestamp(cands["date"].iloc[0])
    ref = ref.assign(date=pd.to_datetime(ref["date"]), price_in=ref["price_in"].astype(float),
                     price_out=ref["price_out"].astype(float))
    cands = cands.assign(price_in=cands["price_in"].astype(float), price_out=cands["price_out"].astype(float))
    r = ref.assign(gap=(ref["date"] - date).abs().dt.days).sort_values("gap")
    r = r[r["gap"] <= max_days]
    if r.empty:
        return None
    rin, rout = r["price_in"].iloc[0], r["price_out"].iloc[0]
    dist = (np.log(cands["price_in"] / rin)).abs() + (np.log(cands["price_out"] / rout)).abs()
    i = dist.idxmin()
    return cands.loc[i] if dist[i] <= 2 * tol else None


@dataclass
class CleanResult:
    rows: pd.DataFrame        # every raw row with classification + drop_reason
    obs: pd.DataFrame         # one row per key x market x date (chosen)
    log: dict


def select_observations(rows: pd.DataFrame, keep_tier_guess: bool = False) -> CleanResult:
    rows = rows.copy()
    cand = rows[rows["drop_reason"] == ""].copy()

    # one host per model key: the maker itself if it sells the model, else the host with most rows
    def _primary_host(g):
        if (g["host"] == g["maker"]).any():
            return g["maker"].iloc[0]
        return g["host"].value_counts().index[0]
    prim = cand.groupby("key").apply(_primary_host, include_groups=False)
    other = cand["host"] != cand["key"].map(prim)
    rows.loc[cand.index[other], "drop_reason"] = "other_host"
    cand = cand[~other]

    # preferred region: per key x source x market keep only the best region rank seen
    # (stops LiteLLM "us/" premium entries or Alibaba "Global" deployments entering the series)
    best_rank = cand.groupby(["key", "source", "market"])["region_rank"].transform("min")
    worse = cand["region_rank"] > best_rank
    rows.loc[cand.index[worse], "drop_reason"] = "non_preferred_region"
    cand = cand[~worse]

    # exact-alias rule: per key x source, if any exact-alias rows exist use only those
    has_exact = cand.groupby(["key", "source"])["exact_alias"].transform("any")
    variant = has_exact & ~cand["exact_alias"]
    rows.loc[cand.index[variant], "drop_reason"] = "dated_variant_of_key"
    cand = cand[~variant]

    keys = ["key", "market", "date", "source"]
    picked, log = [], {"unknown_tier_resolved": 0, "unknown_tier_unresolved": 0}
    std = cand[cand["tier"] == "standard"]
    std_choice = []
    for _, g in std.groupby(keys, sort=False):
        std_choice.append(_pick_row(g))
    std_choice = pd.DataFrame(std_choice)
    picked.append(std_choice.assign(tier_resolution="labelled"))

    unk = cand[cand["tier"] == "unknown"]
    have_std = set(map(tuple, std[keys].values)) if not std.empty else set()
    ref_pool = std_choice[["key", "market", "date", "price_in", "price_out"]].copy() \
        if not std_choice.empty else pd.DataFrame(columns=["key", "market", "date", "price_in", "price_out"])
    for k, g in sorted(unk.groupby(keys, sort=False), key=lambda kv: kv[0][2]):
        if k in have_std:
            continue  # a labelled standard row exists for the same cell
        ref = ref_pool[(ref_pool["key"] == k[0]) & (ref_pool["market"] == k[1])]
        r = _resolve_unknown(g, ref)
        if r is not None:
            log["unknown_tier_resolved"] += 1
            picked.append(r.to_frame().T.assign(tier_resolution="matched_to_labelled"))
            ref_pool = pd.concat([ref_pool, r.to_frame().T[ref_pool.columns]], ignore_index=True)
        else:
            log["unknown_tier_unresolved"] += 1
            if keep_tier_guess:
                picked.append(_pick_row(g).to_frame().T.assign(tier_resolution="guess_median"))
            else:
                rows.loc[g.index, "drop_reason"] = "tier_unresolved"
    picked = [x for x in picked if not x.empty]
    if not picked:
        log.update(rows_in=len(rows), observations=0,
                   drop_reasons=rows["drop_reason"].replace("", "kept_candidate").value_counts().to_dict())
        return CleanResult(rows=rows, obs=pd.DataFrame(columns=list(rows.columns) + ["tier_resolution"]), log=log)
    obs = pd.concat(picked, ignore_index=True)
    obs["date"] = pd.to_datetime(obs["date"])
    for c in ["price_in", "price_out", "price_cached_in", "price_in_local", "price_out_local"]:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")

    # one row per key x market x date: W/Live beat L
    obs["src_rank"] = obs["source"].map({"W": 0, "Live": 0, "L": 1}).fillna(2)
    obs = obs.sort_values(["key", "market", "date", "src_rank", "source"])
    obs = obs.drop_duplicates(["key", "market", "date"], keep="first").drop(columns="src_rank")
    log["rows_in"] = len(rows)
    log["drop_reasons"] = rows["drop_reason"].replace("", "kept_candidate").value_counts().to_dict()
    log["observations"] = len(obs)
    return CleanResult(rows=rows, obs=obs.reset_index(drop=True), log=log)


# ---------------------------------------------------------------------------- monthly panel
def removal_dates(rows: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Date LiteLLM stopped listing the key: last non-suspect removal, provided no LiteLLM
    first_seen/changed for the key comes after it. Uses exact-alias rows when the key has any."""
    lit = rows[rows["source"] == "L"]
    out = {}
    for key, g in lit.groupby("key"):
        if g["exact_alias"].any():
            g = g[g["exact_alias"]]
        rm = g[(g["event"] == "removed") & ~g["is_suspect"]]["date"]
        if rm.empty:
            continue
        later = g[g["event"].isin(["first_seen", "changed", "cache_changed"]) & (g["date"] > rm.max())]
        if later.empty:
            out[key] = rm.max()
    return out


def monthly_panel(res: CleanResult, freq: str = "M", max_carry_months: int = 6,
                  max_carry_months_litellm: int | None = None,
                  market_mode: str = "intl_first", end: str | None = None) -> pd.DataFrame:
    """Snapshot sources (W, Live) are carried forward at most `max_carry_months`. LiteLLM is a
    change log, so its last price stays in force until the next event or removal
    (optionally capped by `max_carry_months_litellm`)."""
    obs = res.obs.copy()
    if obs.empty:
        return pd.DataFrame()
    if market_mode == "intl_first":
        has_intl = obs.groupby("key")["market"].transform(lambda m: (m == "INTL").any())
        obs = obs[(obs["market"] == "INTL") | ~has_intl]
    elif market_mode in ("INTL", "CN"):
        obs = obs[obs["market"] == market_mode]
    else:
        raise ValueError("market_mode must be intl_first, INTL or CN")

    removed = removal_dates(res.rows)
    last_period = pd.Period(end, freq) if end else obs["date"].max().to_period(freq)
    out = []
    for (key, market), g in obs.groupby(["key", "market"]):
        g = g.sort_values("date")
        primary = g[g["source"] != "L"]
        # LiteLLM: only first_seen / changed move list prices
        lit = g[g["source"] == "L"]
        lit_price = lit[lit["event"].isin(["first_seen", "changed"])]
        ev = pd.concat([primary, lit_price]).sort_values("date")
        if ev.empty:
            continue
        cached = g.dropna(subset=["price_cached_in"])[["date", "price_cached_in"]]
        rm = removed.get(key)
        if rm is not None and rm < g["date"].max():
            rm = None  # seen again after removal
        periods = pd.period_range(ev["date"].min().to_period(freq), last_period, freq=freq)
        for per in periods:
            pend = per.end_time.normalize()
            if rm is not None and pend > rm:
                break
            past = ev[ev["date"] <= pend]
            if past.empty:
                continue
            o = past.iloc[-1]
            carry = (per - o["date"].to_period(freq)).n
            cap = max_carry_months if o["source"] != "L" else max_carry_months_litellm
            if cap is not None and carry > cap:
                continue
            c_past = cached[cached["date"] <= pend]
            out.append({
                "model_id": key, "date": pend, "period": str(per), "market": market,
                "maker": o["maker"], "host": o["host"], "third_party_host": bool(o["third_party_host"]),
                "price_in_usd_per_mtok": o["price_in"], "price_out_usd_per_mtok": o["price_out"],
                "price_in_local_per_mtok": o["price_in_local"], "price_out_local_per_mtok": o["price_out_local"],
                "currency_local": o["currency"] or "USD",
                "price_cached_in_usd_per_mtok": c_past["price_cached_in"].iloc[-1] if len(c_past) else np.nan,
                "price_obs_date": o["date"], "price_source": o["source"], "price_event": o["event"],
                "months_since_obs": carry, "is_promo": bool(o["is_promo"]),
                "tier_resolution": o["tier_resolution"], "published_name": o["published"],
                "section": o["section"],
            })
    p = pd.DataFrame(out)
    if p.empty:
        return p
    p["price_blended_3to1_usd_per_mtok"] = (3 * p["price_in_usd_per_mtok"] + p["price_out_usd_per_mtok"]) / 4
    p["log_price_blended"] = np.log(p["price_blended_3to1_usd_per_mtok"])
    # price changes are measured in the currency the provider sets prices in, so RMB list
    # prices do not "change" just because the CNY/USD rate moved
    p["price_blended_3to1_local"] = (3 * p["price_in_local_per_mtok"] + p["price_out_local_per_mtok"]) / 4
    p = p.sort_values(["model_id", "date"]).reset_index(drop=True)
    prev = p.groupby("model_id")["price_blended_3to1_local"].shift()
    prev_cur = p.groupby("model_id")["currency_local"].shift()
    p["price_change_pct"] = np.where(prev_cur == p["currency_local"],
                                     p["price_blended_3to1_local"] / prev - 1, np.nan)
    # "-latest" style keys are pointers that get re-aimed at newer models: their price changes are
    # model swaps. Rises above +300% on one model are almost always a source fixing an earlier error.
    p["is_alias_pointer"] = p["model_id"].str.contains(r"latest$|^chat-latest$", regex=True)
    # exact x10 / x1000 jumps (either direction) are per-1K vs per-1M unit slips
    lg = np.log10(1 + p["price_change_pct"])
    unit_slip = (lg.abs() > 0.5) & ((lg - lg.round()).abs() < 0.02)
    p["suspect_change"] = (p["price_change_pct"] > 3.0) | unit_slip
    p["price_cut"] = (p["price_change_pct"] < -0.005).astype(int)
    p["price_rise"] = (p["price_change_pct"] > 0.005).astype(int)
    first = p.groupby("model_id")["date"].transform("min")
    p["months_since_first_price"] = ((p["date"].dt.year - first.dt.year) * 12
                                     + p["date"].dt.month - first.dt.month)
    p["origin_country"] = p["maker"].map(MAKER_COUNTRY).fillna("")
    return p


# ---------------------------------------------------------------------------- index
def price_index(panel: pd.DataFrame, by: str | None = None, col: str = "price_blended_3to1_local",
                weights: pd.DataFrame | None = None) -> pd.DataFrame:
    """Chained matched-model index (base 100 at first period of each group).

    Alias pointers ("-latest") and suspect changes (> +300%, or exact x10 jumps) are excluded.
    Default column is the local-currency blended price, so the index measures provider
    pricing decisions; pass col="price_blended_3to1_usd_per_mtok" for a USD index that also
    moves with CNY/USD. Relatives across a currency switch are skipped.

    Unweighted: Jevons (geometric mean of price relatives). With `weights`
    (model_id, period, w) the log relatives are weighted by the previous period's w (e.g.
    OpenRouter tokens), which gives a demand-weighted index.
    """
    flags = [c for c in ["is_alias_pointer", "suspect_change"] if c in panel.columns]
    p = panel[["model_id", "period", "date", col, "currency_local"] + flags + ([by] if by else [])].copy()
    p = p.sort_values(["model_id", "date"])
    p["prev"] = p.groupby("model_id")[col].shift()
    p.loc[p.groupby("model_id")["currency_local"].shift() != p["currency_local"], "prev"] = np.nan
    for f in flags:  # alias pointers and suspect jumps never enter the index
        p.loc[p[f].fillna(False).astype(bool), "prev"] = np.nan
    p["prev_period"] = p.groupby("model_id")["period"].shift()
    per = pd.PeriodIndex(p["period"], freq="M")
    prev_per = pd.PeriodIndex(p["prev_period"].fillna(p["period"]), freq="M")
    consec = p["prev"].notna() & (per == prev_per + 1)
    p = p[consec].assign(lr=lambda d: np.log(d[col] / d["prev"]))
    if weights is not None:
        w = weights.rename(columns={"period": "prev_period"})
        w["prev_period"] = w["prev_period"].astype(str)
        p = p.merge(w[["model_id", "prev_period", "w"]], on=["model_id", "prev_period"], how="left")
    else:
        p["w"] = 1.0
    p["w"] = p["w"].fillna(0)
    groups = [by] if by else []
    rel = p.groupby(groups + ["period", "date"]).apply(
        lambda g: pd.Series({"log_rel": np.average(g["lr"], weights=g["w"]) if g["w"].sum() > 0 else np.nan,
                             "n_models": int((g["w"] > 0).sum())}), include_groups=False).reset_index()
    all_periods = panel[groups + ["period", "date"]].drop_duplicates()
    rel = all_periods.merge(rel, on=groups + ["period", "date"], how="left").sort_values(groups + ["date"])
    rel["log_rel"] = rel["log_rel"].fillna(0.0)
    rel["n_models"] = rel["n_models"].fillna(0).astype(int)
    name = f"price_index_{by}" if by else "price_index_all"
    if groups:
        rel[name] = 100 * np.exp(rel.groupby(by)["log_rel"].cumsum())
    else:
        rel[name] = 100 * np.exp(rel["log_rel"].cumsum())
    return rel.drop(columns="log_rel").reset_index(drop=True)


# ---------------------------------------------------------------------------- crosswalk sync
def sync_crosswalk(panel: pd.DataFrame, rows: pd.DataFrame, cw_path: Path) -> tuple[int, int]:
    """Add every priced Part A text model to the crosswalk; fill missing part_a_model_id."""
    from .crosswalk import _norm
    cw = pd.read_csv(cw_path, dtype=str).fillna("")
    # fill part_a_model_id of existing rows when the normalised name matches a Part A key
    keys = set(panel["model_id"])
    filled = 0
    for i, r in cw.iterrows():
        if r["part_a_model_id"] in keys:
            continue
        cands = [r["model_id"].split("/")[-1], r["openrouter_slug"]] + r["aliases"].split("|")
        for cnd in cands:
            k = model_key(cnd) if cnd else ""
            if k in keys:
                cw.at[i, "part_a_model_id"] = k
                filled += 1
                break
        else:
            if r["part_a_model_id"] == r["model_id"]:
                cw.at[i, "part_a_model_id"] = ""  # placeholder from the seed; no Part A match
    taken = {_norm(x) for col in ["model_id", "openrouter_slug", "hf_repo_id", "part_a_model_id"]
             for x in cw[col] if x}
    taken |= {_norm(a) for al in cw["aliases"] for a in al.split("|") if a}
    taken |= {_norm(x.split("/", 1)[1]) for x in cw["model_id"] if "/" in x}
    have = set(cw["part_a_model_id"])
    meta = panel.groupby("model_id").agg(maker=("maker", "first"), first=("date", "min")).reset_index()
    pub = rows[rows["drop_reason"] == ""].groupby("key")["published"].agg(
        lambda s: sorted({x for x in s if x}))
    new = []
    for _, m in meta.iterrows():
        if m["model_id"] in have:
            continue
        mid = f"{m['maker']}/{m['model_id']}"
        if _norm(mid) in taken or _norm(m["model_id"]) in taken:
            continue
        al = []
        for a in pub.get(m["model_id"], []):
            na = _norm(a)
            if na not in taken and "/" not in a and not re.search(r"[\[\(]", a):
                al.append(a)
                taken.add(na)
        taken |= {_norm(mid), _norm(m["model_id"])}
        new.append({"model_id": mid, "provider_id": m["maker"], "model_family": "",
                    "origin_country": MAKER_COUNTRY.get(m["maker"], ""), "open_weights": "",
                    "release_date": "", "openrouter_slug": "", "hf_repo_id": "",
                    "part_a_model_id": m["model_id"], "aliases": "|".join(al[:5])})
    out = pd.concat([cw, pd.DataFrame(new)], ignore_index=True) if new else cw
    out.to_csv(cw_path, index=False)
    return filled, len(new)


# ---------------------------------------------------------------------------- report
def write_report(res: CleanResult, panel: pd.DataFrame, path: Path, sources: list[str]) -> None:
    from ._md import md_table
    rows = res.rows
    dr = rows["drop_reason"].replace("", "kept_candidate").value_counts().rename_axis("reason") \
        .reset_index(name="rows")
    by_src = pd.crosstab(rows["source"], rows["drop_reason"].replace("", "kept_candidate")).reset_index()
    prov = panel.groupby(["maker", "market"]).agg(models=("model_id", "nunique"), model_months=("model_id", "size"),
                                                  first=("period", "min"), last=("period", "max"),
                                                  share_third_party=("third_party_host", "mean"),
                                                  share_carried=("months_since_obs", lambda s: (s > 0).mean())
                                                  ).reset_index().round(2)
    src = panel["price_source"].value_counts().rename_axis("price_source").reset_index(name="model_months")
    tr = panel["tier_resolution"].value_counts().rename_axis("tier_resolution").reset_index(name="model_months")
    chg_cols = ["model_id", "period", "price_change_pct", "price_blended_3to1_usd_per_mtok", "price_source",
                "price_event", "published_name"]
    cuts = panel[panel["price_cut"] == 1].sort_values("price_change_pct").head(25)[chg_cols]
    rises = panel[panel["price_rise"] == 1].sort_values("price_change_pct", ascending=False).head(15)[chg_cols]
    lines = [
        "# Part A price panel: cleaning report", "",
        f"Inputs: {', '.join(sources)}", "",
        f"Raw rows: {len(rows):,}. Chosen observations (model x market x date): {len(res.obs):,}. "
        f"Model-months in panel: {len(panel):,} across {panel['model_id'].nunique()} models, "
        f"{panel['period'].min()} to {panel['period'].max()}.", "",
        f"Unlabelled-tier groups (archived OpenAI 2026 pages): {res.log['unknown_tier_resolved']} resolved "
        f"against a labelled standard price, {res.log['unknown_tier_unresolved']} unresolved.", "",
        "## Row outcomes", "", md_table(dr), "", md_table(by_src), "",
        "## Coverage by maker and market", "", md_table(prov), "",
        "## Where each model-month price comes from", "", md_table(src), "", md_table(tr), "",
        "## Largest month-on-month cuts (check these by hand)", "",
        md_table(cuts.assign(price_change_pct=cuts["price_change_pct"].round(3))), "",
        "## Largest month-on-month rises (often a source correcting an earlier error: check)", "",
        md_table(rises.assign(price_change_pct=rises["price_change_pct"].round(3))), "",
        "## Rules", "",
        "- Standard on-demand tier, base context-length tier, non-thinking mode where both exist.",
        "- Text models only (vision/OCR/omni, audio, image, embedding, moderation and tool endpoints excluded).",
        "- Market: model uses INTL prices if it has any, else CN (RMB converted at the workbook's monthly FX).",
        "- Region order within INTL: International/Singapore, then US, then Global deployments, then Hong Kong.",
        "- Source order on the same date: archived page (W) and live page over LiteLLM (L).",
        "- LiteLLM only moves prices on first_seen/changed events; cache_changed updates cached input only.",
        "- Archived/live snapshots carried forward at most 6 months; a LiteLLM price stays in force until "
        "its next change event or removal (LiteLLM is a change log). Never past a LiteLLM removal.",
        "- Blended price = (3 x input + 1 x output) / 4 per 1M tokens.",
        "- Price changes use the local list currency (RMB prices do not move with the exchange rate).",
        "- Index: chained matched-model Jevons on the local blended price; '-latest' alias pointers and "
        "rises above +300% or exact x10 jumps either way (`suspect_change`: a source correcting itself or a "
        "per-1K/per-1M slip) are left out of the index but kept in the panel.",
    ]
    path.write_text("\n".join(lines) + "\n")
