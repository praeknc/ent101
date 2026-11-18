"""Long-format observation schema shared by every collector.

One row = one value of one variable for one entity over one period from one
source. Collectors only ever produce this table; panel.py reshapes it.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from . import dictionary, sources

OBS_COLUMNS = [
    "obs_id",          # stable hash of the identifying columns
    "source_id",       # key into sources.SOURCES
    "tier",            # A/B/C, copied from the source registry
    "entity_type",     # model | app | provider | package | model_x_app | market
    "entity_id",       # raw id as it appears in the source (slug, repo id, package, provider)
    "model_id",        # canonical model id (crosswalk) or empty
    "provider_id",     # canonical provider id or empty
    "app_id",          # app id for model_x_app rows
    "region",          # global | CN | US | ...
    "period_start",    # first day covered (date)
    "period_end",      # last day covered (date); equals snapshot date for snapshots
    "native_freq",     # D | W | M | Q | irregular | snapshot
    "variable",        # key into dictionary.VARIABLES
    "value",           # float
    "unit",
    "is_estimate",     # True when second-hand, rounded ("over 1 trillion") or lower-bound
    "retrieved_at",    # UTC timestamp of collection
    "source_url",
    "note",            # quote / page / query string / caveat
]

# retrieved_at is deliberately NOT part of the id: re-collecting the same entity x period x
# variable replaces the old value (latest retrieval wins) instead of double counting it.
_ID_COLS = ["source_id", "entity_type", "entity_id", "app_id", "region",
            "period_start", "period_end", "variable"]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_obs(rows: list[dict]) -> pd.DataFrame:
    """Fill defaults, attach tier/unit from the registries and validate."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=OBS_COLUMNS)
    unknown_var = set(df["variable"]) - set(dictionary.VARIABLES)
    if unknown_var:
        raise ValueError(f"Unknown variable(s): {unknown_var}. Add them to the dictionary first.")
    unknown_src = set(df["source_id"]) - set(sources.SOURCES)
    if unknown_src:
        raise ValueError(f"Unknown source_id(s): {unknown_src}")
    defaults = {"model_id": "", "provider_id": "", "app_id": "", "region": "global",
                "is_estimate": False, "retrieved_at": now_utc(), "source_url": "", "note": "",
                "native_freq": None, "unit": None, "tier": None}
    for col, val in defaults.items():
        if col not in df:
            df[col] = val
        elif val is not None:
            df[col] = df[col].fillna(val)
    df["tier"] = df["tier"].where(df["tier"].notna() & (df["tier"] != ""),
                                  df["source_id"].map(sources.tier_of))
    df["unit"] = df["unit"].where(df["unit"].notna() & (df["unit"] != ""),
                                  df["variable"].map(lambda v: dictionary.get(v).unit))
    df["native_freq"] = df["native_freq"].where(
        df["native_freq"].notna() & (df["native_freq"] != ""),
        df["variable"].map(lambda v: dictionary.get(v).native_freq))
    df["obs_id"] = df.apply(_hash_row, axis=1)
    return validate(df[OBS_COLUMNS])


def _hash_row(r) -> str:
    def fmt(v):
        return v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else str(v)
    key = "|".join(fmt(r.get(c, "")) for c in _ID_COLS)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in OBS_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"Observation table missing columns: {missing}")
    df = df.copy()
    for col in ["model_id", "provider_id", "app_id", "entity_id", "source_url", "note"]:
        df[col] = df[col].fillna("").astype(str)
    unknown_src = set(df["source_id"]) - set(sources.SOURCES)
    if unknown_src:
        raise ValueError(f"Unknown source_id(s): {unknown_src}")
    unknown_var = set(df["variable"]) - set(dictionary.VARIABLES)
    if unknown_var:
        raise ValueError(f"Unknown variable(s): {unknown_var}. Add them to the dictionary first.")
    derived = [v for v in set(df["variable"]) if dictionary.get(v).derived]
    if derived:
        raise ValueError(f"Derived variables must not be collected directly: {derived}")
    df["period_start"] = pd.to_datetime(df["period_start"]).dt.normalize()
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.normalize()
    bad = df["period_end"] < df["period_start"]
    if bad.any():
        raise ValueError(f"{bad.sum()} rows have period_end < period_start")
    df["value"] = pd.to_numeric(df["value"], errors="raise").astype(float)
    df["is_estimate"] = df["is_estimate"].astype(str).str.lower().isin(["true", "1", "yes"])
    if df["tier"].isna().any() or not set(df["tier"]).issubset({"A", "B", "C"}):
        raise ValueError("tier must be A, B or C")
    return df


def read_obs(path) -> pd.DataFrame:
    df = validate(pd.read_csv(path, dtype={"model_id": str, "provider_id": str, "app_id": str,
                                           "entity_id": str, "note": str, "source_url": str}))
    # recompute ids so stores written under an older id rule de-duplicate correctly
    df["obs_id"] = df.apply(_hash_row, axis=1)
    return df


def append_obs(new: pd.DataFrame, path) -> pd.DataFrame:
    """Append to a CSV store, de-duplicating on obs_id (latest retrieval wins for identical keys)."""
    from pathlib import Path
    path = Path(path)
    if path.exists():
        old = read_obs(path)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined["retrieved_at"] = combined["retrieved_at"].astype(str)
    combined = combined.sort_values("retrieved_at", kind="stable").drop_duplicates("obs_id", keep="last").sort_values(
        ["source_id", "variable", "entity_id", "period_end"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out = combined.copy()
    for c in ["period_start", "period_end"]:
        out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return combined
