"""OpenRouter: token usage by model and by app.

OpenRouter publishes usage on its rankings pages but has no documented
public API for historical token volumes. This module therefore:

  * `load_export(path)` reads a CSV you build from the rankings page
    (manual transcription, a saved export, or a dataset shared by OpenRouter),
  * `fetch_models()` pulls the official model list (slugs, context, list
    prices) to keep the crosswalk complete and to cross-check Part A prices.

Expected export columns (one row per period x model [x app]):
    period_start  YYYY-MM-DD (first day of the week or month)
    period_end    YYYY-MM-DD (optional; defaults to period_start + 6 days)
    model_slug    e.g. anthropic/claude-3.7-sonnet or deepseek/deepseek-r1:free
    app           optional app name; blank = model total
    tokens        total tokens
    prompt_tokens, completion_tokens   optional
    source_url    optional
"""
from __future__ import annotations

import re

import pandas as pd

from ..crosswalk import Crosswalk, variant_flag
from ..schema import make_obs
from ._http import get_json

MODELS_URL = "https://openrouter.ai/api/v1/models"


def _app_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")


def load_export(path, cw: Crosswalk, freq: str = "W") -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, dtype={"model_slug": str, "app": str})
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"period_start", "model_slug", "tokens"}
    if not required.issubset(df.columns):
        raise ValueError(f"OpenRouter export needs columns {required}, got {list(df.columns)}")
    df["period_start"] = pd.to_datetime(df["period_start"])
    if "period_end" not in df or df["period_end"].isna().all():
        span = pd.Timedelta(days=6) if freq == "W" else None
        df["period_end"] = (df["period_start"] + span) if span is not None else \
            df["period_start"] + pd.offsets.MonthEnd(0)
    df["app"] = df.get("app", pd.Series("", index=df.index)).fillna("")
    mapped, unmatched = cw.resolve(df["model_slug"])
    df["model_id"] = mapped
    df["variant"] = df["model_slug"].map(variant_flag)

    rows = []
    for _, r in df.iterrows():
        base = dict(source_id="openrouter", region="global",
                    period_start=r["period_start"], period_end=r["period_end"], native_freq=freq,
                    model_id=r["model_id"], provider_id=cw.provider_of(r["model_id"]),
                    source_url=r.get("source_url", "") or "https://openrouter.ai/rankings",
                    note=f"variant={r['variant']}" if r["variant"] else "")
        if r["app"]:
            app = _app_id(r["app"])
            rows.append({**base, "entity_type": "model_x_app",
                         "entity_id": f"{app}|{r['model_slug']}", "app_id": app,
                         "variable": "or_app_tokens", "value": r["tokens"]})
        else:
            rows.append({**base, "entity_type": "model", "entity_id": r["model_slug"],
                         "variable": "or_tokens", "value": r["tokens"]})
            for col, var in [("prompt_tokens", "or_tokens_prompt"),
                             ("completion_tokens", "or_tokens_completion")]:
                if col in df and pd.notna(r.get(col)):
                    rows.append({**base, "entity_type": "model", "entity_id": r["model_slug"],
                                 "variable": var, "value": r[col]})
    return make_obs(rows), unmatched


def fetch_models() -> pd.DataFrame:
    """Official model list with list prices (USD per token). Snapshot; store with retrieval date."""
    data = get_json(MODELS_URL)["data"]
    out = pd.DataFrame([{
        "openrouter_slug": m.get("id"),
        "name": m.get("name"),
        "created": pd.to_datetime(m.get("created"), unit="s", errors="coerce"),
        "context_length": m.get("context_length"),
        "price_prompt_usd_per_token": (m.get("pricing") or {}).get("prompt"),
        "price_completion_usd_per_token": (m.get("pricing") or {}).get("completion"),
    } for m in data])
    out["retrieved_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    return out
