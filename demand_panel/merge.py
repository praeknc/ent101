"""Link demand panels to Part A (prices and price indices) by model/provider and date.

Part A input contract (CSV or DataFrame)
----------------------------------------
prices:  one row per model x date (date = day the price was observed or took effect)
    model_id   Part A model identifier (mapped via crosswalk column part_a_model_id)
    date       YYYY-MM-DD
    <price columns>  any numeric columns, e.g. price_in_usd_per_mtok, price_out_usd_per_mtok,
                     price_blended_usd_per_mtok, quality_index
index:   one row per date [x key]   (e.g. a quality-adjusted price index)
    date
    optional key columns (provider_id, segment, region ...)
    <index columns>

Two date rules
--------------
asof         price in force at the END of the demand period (last observation on or before
             period end, within `tolerance_days`). Default: matches "price faced this period".
period_mean  mean of Part A observations falling inside the demand period (needs dense prices).
Both also add `<col>_lag1` (previous period's value) so regressions can use lagged prices.
"""
from __future__ import annotations

import pandas as pd

from .crosswalk import Crosswalk


def _period_end(panel: pd.DataFrame) -> pd.Series:
    per = panel["period"]
    if not isinstance(per.dtype, pd.PeriodDtype):
        per = per.astype(str).map(pd.Period)
    return per.map(lambda p: p.end_time.normalize())


def _value_cols(df: pd.DataFrame, exclude) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def merge_prices(panel: pd.DataFrame, prices: pd.DataFrame, cw: Crosswalk, how: str = "asof",
                 tolerance_days: int = 92, key: str = "model_id") -> tuple[pd.DataFrame, dict]:
    """Attach Part A model prices to a model x period panel. Returns (merged, diagnostics)."""
    panel = panel.copy()
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    pcols = _value_cols(prices, {key, "date"})
    panel["_part_a_id"] = panel[key].map(cw.part_a_id)
    panel["_date"] = _period_end(panel)
    prices = prices.rename(columns={key: "_part_a_id"})

    if how == "asof":
        left = panel.sort_values("_date")
        right = prices.sort_values("date")
        merged = pd.merge_asof(left, right, left_on="_date", right_on="date", by="_part_a_id",
                               direction="backward", tolerance=pd.Timedelta(days=tolerance_days))
        merged = merged.rename(columns={"date": "part_a_price_date"})
    elif how == "period_mean":
        per = panel["period"].iloc[0].freqstr if len(panel) else "M"
        prices["period"] = prices["date"].dt.to_period(per)
        pm = prices.groupby(["_part_a_id", "period"], as_index=False)[pcols].mean()
        merged = panel.merge(pm, on=["_part_a_id", "period"], how="left")
    else:
        raise ValueError("how must be 'asof' or 'period_mean'")

    merged = merged.sort_values([key, "period"])
    for c in pcols:
        merged[f"{c}_lag1"] = merged.groupby(key)[c].shift()

    demand_ids = set(panel["_part_a_id"])
    price_ids = set(prices["_part_a_id"])
    matched_rows = merged[pcols[0]].notna() if pcols else pd.Series(False, index=merged.index)
    diag = {
        "rows": len(merged),
        "rows_with_price": int(matched_rows.sum()),
        "row_match_rate": float(matched_rows.mean()) if len(merged) else 0.0,
        "models_in_demand_not_in_part_a": sorted(demand_ids - price_ids),
        "models_in_part_a_not_in_demand": sorted(price_ids - demand_ids),
    }
    return merged.drop(columns=["_date"]).reset_index(drop=True), diag


def merge_index(panel: pd.DataFrame, index: pd.DataFrame, by: list[str] | None = None,
                tolerance_days: int = 92) -> tuple[pd.DataFrame, dict]:
    """Attach a Part A price index (as of period end), optionally matched on key columns."""
    by = by or []
    index = index.copy()
    index["date"] = pd.to_datetime(index["date"]).dt.normalize()
    icols = _value_cols(index, set(by) | {"date"})
    left = panel.copy()
    left["_date"] = _period_end(left)
    merged = pd.merge_asof(left.sort_values("_date"), index.sort_values("date"),
                           left_on="_date", right_on="date", by=by or None,
                           direction="backward", tolerance=pd.Timedelta(days=tolerance_days))
    merged = merged.rename(columns={"date": "part_a_index_date"}).drop(columns="_date")
    sort_keys = [c for c in merged.columns if c.endswith("_id")][:1] + ["period"]
    merged = merged.sort_values(sort_keys)
    grp = sort_keys[0] if len(sort_keys) > 1 else None
    for c in icols:
        merged[f"{c}_lag1"] = merged.groupby(grp)[c].shift() if grp else merged[c].shift()
    ok = merged[icols[0]].notna() if icols else pd.Series(False, index=merged.index)
    return merged.reset_index(drop=True), {"rows": len(merged), "rows_with_index": int(ok.sum()),
                                           "row_match_rate": float(ok.mean()) if len(merged) else 0.0}


def rollup_prices_to_provider(prices: pd.DataFrame, cw: Crosswalk, weights: pd.DataFrame | None = None,
                              freq: str = "M") -> pd.DataFrame:
    """Provider x period price (simple or demand-weighted mean of model prices) for provider panels.

    weights: optional model_id x period frame with column `w` (e.g. or_tokens) for weighting.
    """
    pr = prices.copy()
    pr["date"] = pd.to_datetime(pr["date"])
    pcols = _value_cols(pr, {"model_id", "date"})
    inv = {cw.part_a_id(m): m for m in cw.models["model_id"]}
    pr["model_id"] = pr["model_id"].map(lambda x: inv.get(x, x))
    pr["provider_id"] = pr["model_id"].map(cw.provider_of)
    pr["period"] = pr["date"].dt.to_period(freq)
    pm = pr.sort_values("date").groupby(["provider_id", "model_id", "period"], as_index=False)[pcols].last()
    if weights is not None:
        pm = pm.merge(weights[["model_id", "period", "w"]], on=["model_id", "period"], how="left")
        pm["w"] = pm["w"].fillna(0)
        out = pm.groupby(["provider_id", "period"]).apply(
            lambda g: pd.Series({c: (g[c] * g["w"]).sum() / g["w"].sum() if g["w"].sum() > 0
                                 else g[c].mean() for c in pcols}), include_groups=False)
        return out.reset_index()
    return pm.groupby(["provider_id", "period"], as_index=False)[pcols].mean()
