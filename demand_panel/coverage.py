"""Coverage and gap reporting. The thesis must state the gaps, so this writes them out."""
from __future__ import annotations

import pandas as pd

from ._md import md_table

from . import sources
from .crosswalk import Crosswalk


def source_summary(obs: pd.DataFrame, freq: str = "M", start: str | None = None,
                   end: str | None = None) -> pd.DataFrame:
    if obs.empty:
        return pd.DataFrame()
    o = obs.assign(period=obs["period_end"].dt.to_period(freq))
    g = o.groupby("source_id")
    out = pd.DataFrame({
        "tier": g["tier"].first(),
        "first_period": g["period"].min().astype(str),
        "last_period": g["period"].max().astype(str),
        "n_periods": g["period"].nunique(),
        "n_entities": g["entity_id"].nunique(),
        "n_obs": g.size(),
        "share_estimate": g["is_estimate"].mean().round(3),
    })
    all_periods = pd.period_range(start or o["period"].min(), end or o["period"].max(), freq=freq)
    # share of periods in the report window with any observation
    out["period_coverage"] = (out["n_periods"] / len(all_periods)).round(3)
    for s in sources.SOURCES:
        if s not in out.index:
            out.loc[s] = [sources.get(s).tier, "", "", 0, 0, 0, None, 0.0]
    return out.sort_values(["tier"]).reset_index()


def model_source_matrix(obs: pd.DataFrame, cw: Crosswalk, freq: str = "M") -> pd.DataFrame:
    """For each crosswalk model: number of periods observed in each source (0 = never)."""
    o = obs[obs["model_id"] != ""].assign(period=obs["period_end"].dt.to_period(freq))
    mat = o.groupby(["model_id", "source_id"])["period"].nunique().unstack(fill_value=0)
    mat = mat.reindex(cw.models["model_id"], fill_value=0)
    meta = cw.models.set_index("model_id")[["provider_id", "origin_country", "open_weights"]]
    return meta.join(mat).fillna(0).reset_index()


def china_gaps(obs: pd.DataFrame, cw: Crosswalk, start: str, end: str, freq: str = "M") -> pd.DataFrame:
    """CN-origin providers x period cells with no Chinese demand observation."""
    cn_prov = sorted(set(cw.models.loc[cw.models["origin_country"] == "CN", "provider_id"]))
    periods = pd.period_range(start, end, freq=freq)
    grid = pd.MultiIndex.from_product([cn_prov, periods], names=["provider_id", "period"]).to_frame(index=False)
    cn = obs[(obs["source_id"] == "china_demand")].assign(period=lambda d: d["period_end"].dt.to_period(freq))
    have = cn.groupby(["provider_id", "period"]).size().rename("n_cn_obs").reset_index()
    grid = grid.merge(have, how="left", on=["provider_id", "period"]).fillna({"n_cn_obs": 0})
    other = obs[obs["provider_id"].isin(cn_prov)].assign(period=lambda d: d["period_end"].dt.to_period(freq))
    oth = other[other["source_id"] != "china_demand"].groupby(["provider_id", "period"])["source_id"] \
        .agg(lambda s: "|".join(sorted(set(s)))).rename("other_sources").reset_index()
    grid = grid.merge(oth, how="left", on=["provider_id", "period"]).fillna({"other_sources": ""})
    grid["gap"] = grid["n_cn_obs"] == 0
    return grid


def write_report(obs: pd.DataFrame, cw: Crosswalk, path, start: str, end: str, freq: str = "M",
                 unmatched: list[str] | None = None) -> str:
    ss = source_summary(obs, freq, start, end)
    mm = model_source_matrix(obs, cw, freq)
    cg = china_gaps(obs, cw, start, end, freq)
    lines = ["# Demand panel coverage report", "",
             f"Window {start} to {end}, frequency {freq}. Generated from {len(obs):,} observations.", "",
             "## By source", "", md_table(ss) if not ss.empty else "_no observations_", "",
             "## Periods observed per model and source", "", md_table(mm), "",
             "## Chinese demand gaps", ""]
    if cg.empty:
        lines.append("_No CN-origin models in crosswalk._")
    else:
        summ = cg.groupby("provider_id").agg(periods=("gap", "size"), gaps=("gap", "sum")).reset_index()
        summ["gap_share"] = (summ["gaps"] / summ["periods"]).round(2)
        lines += [md_table(summ), "",
                  "Cells with no Chinese-source observation but some other source "
                  "(the only evidence on CN demand in that period is non-Chinese):", ""]
        alt = cg[cg["gap"] & (cg["other_sources"] != "")]
        lines.append(md_table(alt.assign(period=alt["period"].astype(str)))
                     if not alt.empty else "_none_")
    if unmatched:
        lines += ["", "## Ids not in crosswalk (dropped from model panel)", ""] + [f"- `{u}`" for u in unmatched]
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text
