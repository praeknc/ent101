"""Turn the long observation table into analysis panels.

Three panels, matching the three units of analysis in 4.4:

  model_panel     model x period      (OpenRouter tokens/shares, HF downloads, CN ranks)
  provider_panel  provider x period   (SDK downloads, GitHub repos, disclosures, CN announcements,
                                       plus model-level flows summed to the provider)
  app_panel       app x model x period (OpenRouter app routing: who switches to what)

Each builder returns (wide, long) where `long` keeps quality columns
(n_obs, sources, min tier, any estimate) and `wide` is one column per variable.
Missing means "not observed", never zero.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import dictionary, sources

TIER_ORDER = {"A": 0, "B": 1, "C": 2}
_PRORATE_FREQS = {"D", "W", "M"}


# ---------------------------------------------------------------------------- helpers
def _exclude_validation(obs: pd.DataFrame, include_validation: bool) -> pd.DataFrame:
    if include_validation:
        return obs
    keep = ~obs["source_id"].map(lambda s: sources.get(s).validation_only)
    return obs[keep]


def assign_periods(obs: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Give every row a panel period.

    Flow ('sum') rows from regular native frequencies that straddle a period
    boundary (e.g. a week spanning two months) are split pro rata by days.
    Everything else is assigned to the period containing period_end.
    """
    obs = obs.copy()
    rules = obs["variable"].map(dictionary.agg_rule)
    span = (obs["period_end"] - obs["period_start"]).dt.days + 1
    p0 = obs["period_start"].dt.to_period(freq)
    p1 = obs["period_end"].dt.to_period(freq)
    split = (rules == "sum") & obs["native_freq"].isin(_PRORATE_FREQS) & (p0 != p1)

    simple = obs[~split].copy()
    simple["period"] = p1[~split]
    simple["weight"] = 1.0

    pieces = [simple]
    for idx, r in obs[split].iterrows():
        for per in pd.period_range(p0[idx], p1[idx], freq=freq):
            a = max(r["period_start"], per.start_time.normalize())
            b = min(r["period_end"], per.end_time.normalize())
            days = (b - a).days + 1
            if days <= 0:
                continue
            row = r.copy()
            row["period"] = per
            row["weight"] = days / span[idx]
            row["value"] = r["value"] * row["weight"]
            pieces.append(row.to_frame().T)
    out = pd.concat(pieces, ignore_index=True)
    out["value"] = out["value"].astype(float)
    return out


def _aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Collapse to keys + period + variable using each variable's agg rule."""
    if df.empty:
        return pd.DataFrame(columns=keys + ["period", "variable", "value", "n_obs", "sources",
                                            "tier", "any_estimate"])
    df = df.sort_values("period_end")
    gcols = keys + ["period", "variable"]
    out = []
    for rule, sub in df.groupby(df["variable"].map(dictionary.agg_rule)):
        g = sub.groupby(gcols, sort=False)
        val = getattr(g["value"], {"sum": "sum", "last": "last", "mean": "mean", "max": "max"}[rule])()
        meta = g.agg(n_obs=("value", "size"),
                     sources=("source_id", lambda s: "|".join(sorted(set(s)))),
                     tier=("tier", lambda t: min(t, key=TIER_ORDER.get)),
                     any_estimate=("is_estimate", "any"))
        out.append(meta.assign(value=val).reset_index())
    return pd.concat(out, ignore_index=True)


def _wide(long: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame(columns=keys + ["period"])
    w = long.pivot_table(index=keys + ["period"], columns="variable", values="value", aggfunc="first")
    w.columns.name = None
    return w.reset_index().sort_values(keys + ["period"]).reset_index(drop=True)


def _add_derived(long: pd.DataFrame, keys: list[str], rows: list) -> pd.DataFrame:
    cols = keys + ["period", "variable", "value", "n_obs", "sources", "tier", "any_estimate"]
    filled = []
    for d in rows:
        if d is None or d.empty:
            continue
        d = d.copy()
        for c, default in [("n_obs", 1), ("sources", "derived"), ("tier", "B"), ("any_estimate", False)]:
            d[c] = d[c].fillna(default) if c in d else default
        filled.append(d[cols])
    return pd.concat([long] + filled, ignore_index=True) if filled else long


def _aggregate_with_repo_rollup(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Stocks/snapshots ('last') can come from several raw entities per key (e.g. the official
    HF repo and a re-upload of the same model). Take the last value per raw entity, then sum
    across entities. Flows/means/max aggregate directly."""
    is_last = rows["variable"].map(dictionary.agg_rule) == "last"
    flow = _aggregate(rows[~is_last], keys)
    per_ent = _aggregate(rows[is_last], keys + ["entity_id"])
    if per_ent.empty:
        return flow
    stock = per_ent.groupby(keys + ["period", "variable"], as_index=False).agg(
        value=("value", "sum"), n_obs=("n_obs", "sum"),
        sources=("sources", lambda s: "|".join(sorted(set("|".join(s).split("|"))))),
        tier=("tier", "min"), any_estimate=("any_estimate", "any"))
    return pd.concat([flow, stock], ignore_index=True)


def _period_last_per_entity(obs: pd.DataFrame, entity: str, var: str) -> pd.DataFrame:
    s = obs[obs["variable"] == var].sort_values("period_end")
    return s.groupby([entity, "entity_id", "period"], as_index=False).last()


def _hf_downloads_period(per_obs: pd.DataFrame, key: str) -> pd.DataFrame:
    """First difference of all-time downloads between consecutive panel periods, per repo,
    then summed to `key`. Gaps of more than one period give NaN (cannot be attributed)."""
    last = _period_last_per_entity(per_obs, key, "hf_downloads_alltime")
    if last.empty:
        return None
    last = last.sort_values(["entity_id", "period"])
    last["prev_val"] = last.groupby("entity_id")["value"].shift()
    last["prev_per"] = last.groupby("entity_id")["period"].shift()
    consecutive = last["prev_per"].notna() & (last["period"] == last["prev_per"] + 1)
    last["d"] = np.where(consecutive, last["value"] - last["prev_val"], np.nan)
    d = last.dropna(subset=["d"]).groupby([key, "period"], as_index=False)["d"].sum(min_count=1)
    return d.rename(columns={"d": "value"}).assign(variable="hf_downloads_period",
                                                   sources="huggingface", tier="A")


# ---------------------------------------------------------------------------- model panel
def build_model_panel(obs: pd.DataFrame, freq: str = "M", include_validation: bool = False):
    obs = _exclude_validation(obs, include_validation)
    m = obs[obs["entity_type"].isin(["model", "model_x_app"])]
    unmatched = sorted(set(m.loc[m["model_id"] == "", "entity_id"]))
    if unmatched:
        warnings.warn(f"{len(unmatched)} model rows without canonical model_id (add to crosswalk): "
                      f"{unmatched[:10]}{' ...' if len(unmatched) > 10 else ''}")
    m = assign_periods(m[m["model_id"] != ""], freq)
    keys = ["model_id"]

    model_rows = m[m["entity_type"] == "model"]
    long = _aggregate_with_repo_rollup(model_rows, keys)

    derived = []
    tok = long[long["variable"] == "or_tokens"]
    if not tok.empty:
        tot = tok.groupby("period")["value"].transform("sum")
        derived.append(tok.assign(variable="or_token_share", value=tok["value"] / tot))
    app = m[m["entity_type"] == "model_x_app"]
    if not app.empty:
        a = app.groupby(["model_id", "period", "app_id"], as_index=False)["value"].sum()
        n_apps = a[a["value"] > 0].groupby(["model_id", "period"], as_index=False)["app_id"].nunique()
        derived.append(n_apps.rename(columns={"app_id": "value"}).assign(
            variable="or_n_apps", sources="openrouter", tier="A"))
        top = a.groupby(["model_id", "period"], as_index=False)["value"].max()
        if not tok.empty:
            top = top.merge(tok[["model_id", "period", "value"]], on=["model_id", "period"],
                            suffixes=("_top", "_tot"))
            top["value"] = np.clip(top["value_top"] / top["value_tot"], 0, 1)
            derived.append(top[["model_id", "period", "value"]].assign(
                variable="or_top_app_share", sources="openrouter", tier="A"))
    derived.append(_hf_downloads_period(model_rows, "model_id"))
    long = _add_derived(long, keys, derived)
    return _wide(long, keys), long


# ---------------------------------------------------------------------------- provider panel
def build_provider_panel(obs: pd.DataFrame, freq: str = "M", include_validation: bool = False,
                         rollup_model_vars=("or_tokens", "hf_downloads_alltime", "hf_downloads_30d")):
    obs = _exclude_validation(obs, include_validation)
    p = assign_periods(obs[obs["provider_id"] != ""], freq)
    keys = ["provider_id"]

    native = p[p["entity_type"] == "provider"]
    long = _aggregate(native, keys)

    derived = []
    pk = p[p["entity_type"] == "package"]
    if not pk.empty:
        sdk = _aggregate(pk, keys)
        sdk = sdk.groupby(keys + ["period"], as_index=False).agg(
            value=("value", "sum"), n_obs=("n_obs", "sum"),
            sources=("sources", lambda s: "|".join(sorted(set("|".join(s).split("|"))))),
            tier=("tier", "min"), any_estimate=("any_estimate", "any"))
        # keep ecosystem-level columns too
        long = pd.concat([long, _aggregate(pk, keys)], ignore_index=True)
        derived.append(sdk.assign(variable="sdk_downloads"))

    # model-level flows summed to provider (useful for provider-level price index merges)
    mv = p[(p["entity_type"] == "model") & p["variable"].isin(rollup_model_vars)]
    if not mv.empty:
        long = pd.concat([long, _aggregate_with_repo_rollup(mv, keys)], ignore_index=True)
        derived.append(_hf_downloads_period(mv, "provider_id"))

    gh = long[long["variable"] == "gh_new_repos"]
    allr = assign_periods(obs[(obs["variable"] == "gh_new_repos_all")], freq)
    if not gh.empty and not allr.empty:
        denom = allr.groupby("period")["value"].sum().rename("denom")
        g = gh.join(denom, on="period").dropna(subset=["denom"])
        derived.append(g.assign(variable="gh_new_repos_share", value=g["value"] / g["denom"])
                       .drop(columns="denom"))
    long = _add_derived(long, keys, derived)
    return _wide(long, keys), long


# ---------------------------------------------------------------------------- app panel
def build_app_panel(obs: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """app x model x period tokens, with the model's share of the app's tokens (the app's 'choice')."""
    a = obs[(obs["variable"] == "or_app_tokens") & (obs["model_id"] != "")]
    if a.empty:
        return pd.DataFrame(columns=["app_id", "model_id", "period", "tokens", "share_in_app"])
    a = assign_periods(a, freq)
    out = a.groupby(["app_id", "model_id", "period"], as_index=False)["value"].sum()
    out = out.rename(columns={"value": "tokens"})
    out["share_in_app"] = out["tokens"] / out.groupby(["app_id", "period"])["tokens"].transform("sum")
    return out.sort_values(["app_id", "period", "tokens"], ascending=[True, True, False]).reset_index(drop=True)
