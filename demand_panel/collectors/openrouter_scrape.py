"""Scrape OpenRouter usage data from the public endpoints behind openrouter.ai/rankings.

These are the site's own frontend endpoints (undocumented, may change without notice):

  /api/frontend/v1/rankings/model-rankings-chart   ~53 weeks x top ~9 models + "Others"
  /api/frontend/v1/rankings/models?view=week       every model variant, current week
  /api/frontend/v1/stats/model-activity            daily tokens per model variant, last ~31 days
  /api/frontend/v1/rankings/apps                   top 20 apps by day / week / month
  /api/v1/models                                   catalog: slug <-> permaslug, list prices

History beyond these windows is not available, so run `collect openrouter` WEEKLY: each run
adds the last ~31 days of daily data for every model, and the store de-duplicates overlaps.

Raw responses are saved to data/raw/openrouter/<YYYYMMDD>/ so every number is traceable.
"""
from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path

import pandas as pd

from ..crosswalk import Crosswalk
from ..schema import make_obs, now_utc
from ._http import get_json

BASE = "https://openrouter.ai"
CHART = BASE + "/api/frontend/v1/rankings/model-rankings-chart"
MODELS_WEEK = BASE + "/api/frontend/v1/rankings/models"
ACTIVITY = BASE + "/api/frontend/v1/stats/model-activity"
APPS = BASE + "/api/frontend/v1/rankings/apps"
CATALOG = BASE + "/api/v1/models"

AUTHOR_TO_PROVIDER = {"qwen": "alibaba", "z-ai": "zhipu", "moonshotai": "moonshot", "meta-llama": "meta",
                      "mistralai": "mistral", "x-ai": "xai", "baidu": "baidu", "deepseek": "deepseek",
                      "minimax": "minimax", "openai": "openai", "anthropic": "anthropic", "google": "google"}
CN_AUTHORS = {"qwen", "alibaba", "deepseek", "z-ai", "zhipu", "moonshotai", "moonshot", "minimax", "baidu",
              "tencent", "xiaomi", "stepfun", "bytedance", "bytedance-seed", "thudm", "01-ai", "meituan",
              "inclusionai", "baichuan", "kwaipilot", "iflytek", "sensetime"}
US_AUTHORS = {"openai", "anthropic", "google", "meta-llama", "meta", "x-ai", "nvidia", "microsoft", "amazon",
              "perplexity", "cohere", "ibm-granite", "liquid", "nousresearch", "allenai"}


def strip_date(permaslug: str) -> str:
    return re.sub(r"-20\d{6}$", "", permaslug)


def provider_of_author(author: str) -> str:
    return AUTHOR_TO_PROVIDER.get(author, author)


def country_of_author(author: str) -> str:
    if author in CN_AUTHORS:
        return "CN"
    if author in US_AUTHORS:
        return "US"
    if author in {"mistralai"}:
        return "FR"
    return ""


# ---------------------------------------------------------------------------- fetch
def fetch_all(raw_dir: Path, sleep: float = 0.25, max_models: int | None = None, log=print) -> Path:
    """Download everything into raw_dir/<YYYYMMDD>/ and return that folder."""
    stamp = now_utc()[:10].replace("-", "")
    out = Path(raw_dir) / stamp
    out.mkdir(parents=True, exist_ok=True)
    for name, url, params in [("chart", CHART, None), ("models_week", MODELS_WEEK, {"view": "week"}),
                              ("models_month", MODELS_WEEK, {"view": "month"}), ("apps", APPS, None),
                              ("catalog", CATALOG, None)]:
        data = get_json(url, params=params)
        (out / f"{name}.json").write_text(json.dumps(data))
        log(f"  {name}: ok")
    week = json.loads((out / "models_week.json").read_text())["data"]
    pairs = sorted({(r["model_permaslug"], r["variant"]) for r in week if r.get("model_permaslug")})
    if max_models:
        pairs = pairs[:max_models]
    acts, failed = {}, []
    for i, (slug, var) in enumerate(pairs, 1):
        try:
            d = get_json(ACTIVITY, params={"permaslug": slug, "variant": var}, retries=3)
            acts[f"{slug}|{var}"] = d.get("data", d).get("analytics", [])
        except Exception as e:  # keep going; record failures
            failed.append(f"{slug}|{var}: {e}")
        if i % 100 == 0:
            log(f"  activity: {i}/{len(pairs)}")
        time.sleep(sleep)
    with gzip.open(out / "activity.json.gz", "wt") as f:
        json.dump(acts, f)
    (out / "meta.json").write_text(json.dumps({"retrieved_at": now_utc(), "n_activity": len(acts),
                                               "failed": failed}, indent=1))
    log(f"  activity: {len(acts)} model variants, {len(failed)} failed")
    return out


# ---------------------------------------------------------------------------- parse
def _load(folder: Path):
    j = lambda n: json.loads((folder / f"{n}.json").read_text())
    with gzip.open(folder / "activity.json.gz", "rt") as f:
        acts = json.load(f)
    meta = j("meta")
    return j("chart"), j("catalog"), j("apps"), acts, meta


def slug_map(catalog: dict) -> dict[str, str]:
    """permaslug -> public slug (falls back to dropping the date suffix). The catalog also lists
    pricing variants (":batch", ":free") under the same permaslug; the plain id wins."""
    out: dict[str, str] = {}
    for m in catalog.get("data", []):
        perma = m.get("canonical_slug") or m["id"]
        if ":" in m["id"] and perma in out:
            continue
        if perma not in out or ":" in out[perma]:
            out[perma] = m["id"]
    return {k: v.split(":")[0] for k, v in out.items()}


def resolve_model(cw: Crosswalk, slug: str, permaslug: str) -> str:
    for raw in (slug, permaslug, strip_date(permaslug)):
        mid = cw.model_id(raw)
        if mid:
            return mid
    return ""


def parse_folder(folder: Path, cw: Crosswalk) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Return (observations, readable weekly table, new crosswalk rows for unknown models)."""
    chart, catalog, apps, acts, meta = _load(folder)
    retrieved = pd.Timestamp(meta["retrieved_at"][:10])
    perma2slug = slug_map(catalog)
    rows, table, new_models, seen_new = [], [], [], set()

    def model_fields(permaslug: str):
        slug = perma2slug.get(permaslug, strip_date(permaslug))
        mid = resolve_model(cw, slug, permaslug)
        if not mid and slug not in seen_new:
            seen_new.add(slug)
            author = slug.split("/")[0]
            new_models.append({"model_id": slug, "provider_id": provider_of_author(author), "model_family": "",
                               "origin_country": country_of_author(author), "open_weights": "",
                               "release_date": "", "openrouter_slug": slug, "hf_repo_id": "",
                               "part_a_model_id": "", "aliases": permaslug})
        mid = mid or slug
        prov = cw.provider_of(mid) or provider_of_author(slug.split("/")[0])
        return slug, mid, prov

    # 1. weekly chart: top models + Others -> or_tokens_topweekly, platform total
    for wk in chart["data"]["data"]:
        start = pd.Timestamp(wk["x"])
        end = start + pd.Timedelta(days=6)
        if end >= retrieved:  # current week is incomplete
            continue
        total = sum(wk["ys"].values())
        rows.append(dict(source_id="openrouter", entity_type="market", entity_id="openrouter_all",
                         period_start=start, period_end=end, native_freq="W",
                         variable="or_tokens_platform", value=total, source_url=CHART,
                         note="sum of chart series incl. Others"))
        for key, val in wk["ys"].items():
            if key == "Others":
                continue
            permaslug, _, variant = key.partition(":")
            slug, mid, prov = model_fields(permaslug)
            rows.append(dict(source_id="openrouter", entity_type="model", entity_id=key, model_id=mid,
                             provider_id=prov, period_start=start, period_end=end, native_freq="W",
                             variable="or_tokens_topweekly", value=val, source_url=CHART,
                             note=f"top-N weekly chart; variant={variant or 'standard'}"))
            table.append(dict(week_start=start.date(), model_slug=slug, permaslug=permaslug,
                              variant=variant or "standard", model_id=mid, tokens=val,
                              prompt_tokens=None, completion_tokens=None, requests=None,
                              source="weekly_chart_topN"))

    # 2. daily activity for every model variant -> or_tokens (+ free subset, requests)
    for key, days in acts.items():
        permaslug, variant = key.split("|")
        slug, mid, prov = model_fields(permaslug)
        for d in days:
            day = pd.Timestamp(d["date"][:10])
            if day >= retrieved:  # today is incomplete
                continue
            tok = (d.get("total_prompt_tokens") or 0) + (d.get("total_completion_tokens") or 0)
            base = dict(source_id="openrouter", entity_type="model", entity_id=f"{permaslug}:{variant}",
                        model_id=mid, provider_id=prov, period_start=day, period_end=day, native_freq="D",
                        source_url=f"{ACTIVITY}?permaslug={permaslug}&variant={variant}",
                        note=f"variant={variant}")
            rows.append({**base, "variable": "or_tokens", "value": tok})
            rows.append({**base, "variable": "or_tokens_prompt", "value": d.get("total_prompt_tokens") or 0})
            rows.append({**base, "variable": "or_tokens_completion",
                         "value": d.get("total_completion_tokens") or 0})
            rows.append({**base, "variable": "or_requests", "value": d.get("count") or 0})
            if variant == "free":
                rows.append({**base, "variable": "or_tokens_free", "value": tok})

    # readable weekly table from the daily data (complete weeks only, Monday starts)
    if acts:
        daily = pd.DataFrame([dict(permaslug=k.split("|")[0], variant=k.split("|")[1], day=pd.Timestamp(d["date"][:10]),
                                   prompt=d.get("total_prompt_tokens") or 0,
                                   completion=d.get("total_completion_tokens") or 0, requests=d.get("count") or 0)
                              for k, v in acts.items() for d in v])
        daily = daily[daily["day"] < retrieved]
        daily["week_start"] = daily["day"] - pd.to_timedelta(daily["day"].dt.weekday, unit="D")
        full = daily.groupby(["permaslug", "variant", "week_start"]).filter(lambda g: g["day"].nunique() == 7)
        wk = full.groupby(["permaslug", "variant", "week_start"], as_index=False)[["prompt", "completion",
                                                                                   "requests"]].sum()
        for _, r in wk.iterrows():
            slug, mid, _ = model_fields(r["permaslug"])
            table.append(dict(week_start=r["week_start"].date(), model_slug=slug, permaslug=r["permaslug"],
                              variant=r["variant"], model_id=mid, tokens=r["prompt"] + r["completion"],
                              prompt_tokens=r["prompt"], completion_tokens=r["completion"],
                              requests=r["requests"], source="daily_activity_all_models"))

    # 3. app totals (snapshot windows ending at retrieval)
    for window, days in [("day", 1), ("week", 7), ("month", 30)]:
        for a in apps["data"].get(window, []):
            app = a.get("app") or {}
            rows.append(dict(source_id="openrouter", entity_type="app", entity_id=app.get("slug") or str(a["app_id"]),
                             app_id=app.get("slug") or str(a["app_id"]),
                             period_start=retrieved - pd.Timedelta(days=days), period_end=retrieved - pd.Timedelta(days=1),
                             native_freq="snapshot", variable=f"or_app_tokens_{window}", value=float(a["total_tokens"]),
                             is_estimate=True, source_url=APPS,
                             note=f"rank={a.get('rank')}; window end approximate; categories="
                                  + ",".join(app.get("categories") or [])))

    obs = make_obs(rows)
    tab = pd.DataFrame(table).sort_values(["week_start", "tokens"], ascending=[True, False]) if table else pd.DataFrame()
    return obs, tab, new_models


def _keys(r: dict) -> set[str]:
    from ..crosswalk import _norm
    ks = [r["model_id"], r["openrouter_slug"], r["hf_repo_id"], r["part_a_model_id"]]
    ks += [a for a in r["aliases"].split("|") if a]
    ks += [k.split("/", 1)[1] for k in ks if "/" in k]
    return {_norm(k) for k in ks if k}


def add_models_to_crosswalk(new_models: list[dict], cw_path: Path) -> int:
    """Append unknown OpenRouter models, skipping any whose names collide (e.g. case-only
    differences) with a model already in the crosswalk."""
    if not new_models:
        return 0
    cw = pd.read_csv(cw_path, dtype=str).fillna("")
    taken: set[str] = set()
    for r in cw.to_dict("records"):
        taken |= _keys(r)
    add = []
    for m in new_models:
        k = _keys(m)
        if k & taken:
            continue
        add.append(m)
        taken |= k
    if add:
        pd.concat([cw, pd.DataFrame(add)], ignore_index=True).to_csv(cw_path, index=False)
    return len(add)
