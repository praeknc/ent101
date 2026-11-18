"""Hand-entered sources: provider disclosures, Chinese demand, external benchmarks.

Each has a CSV template in data/manual/. Rules enforced here:
  * every row needs a source_url and a verbatim quote (or table/page ref) in `quote`,
  * `variable` must be in the dictionary and belong to the file's source,
  * values are stored exactly as stated; unit conversions go in a new row with
    is_estimate=True and the conversion written in `note`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import dictionary
from ..crosswalk import Crosswalk
from ..schema import make_obs

MANUAL_COLUMNS = ["source_id", "provider_id", "model_raw", "region", "period_start", "period_end",
                  "variable", "value", "unit_as_stated", "is_estimate", "source_url", "source_doc",
                  "page_or_timestamp", "quote", "date_published", "note"]

FILES = {
    "provider_disclosures": "provider_disclosures.csv",
    "china_demand": "china_demand.csv",
    "external_benchmarks": "external_benchmarks.csv",
}


def write_templates(manual_dir: Path) -> None:
    manual_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        p = manual_dir / fname
        if not p.exists():
            pd.DataFrame(columns=MANUAL_COLUMNS).to_csv(p, index=False)


def load(path, cw: Crosswalk, source_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    if df.empty:
        return make_obs([])
    missing = set(MANUAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    errors = []
    for i, r in df.iterrows():
        line = i + 2
        if r["source_id"] != source_id:
            errors.append(f"line {line}: source_id '{r['source_id']}' != '{source_id}'")
        if not r["source_url"]:
            errors.append(f"line {line}: source_url is required")
        if not (r["quote"] or r["page_or_timestamp"]):
            errors.append(f"line {line}: quote or page_or_timestamp is required")
        if r["variable"] not in dictionary.VARIABLES:
            errors.append(f"line {line}: unknown variable '{r['variable']}'")
        elif dictionary.get(r["variable"]).source_id != source_id:
            errors.append(f"line {line}: variable '{r['variable']}' belongs to "
                          f"{dictionary.get(r['variable']).source_id}")
    if errors:
        raise ValueError(f"{path} failed validation:\n  " + "\n  ".join(errors))

    rows = []
    for _, r in df.iterrows():
        model_id = cw.model_id(r["model_raw"]) if r["model_raw"] else ""
        level = dictionary.get(r["variable"]).entity_level
        entity_id = (model_id or r["model_raw"]) if level == "model" else (r["provider_id"] or r["region"])
        note = "; ".join(x for x in [
            f"doc={r['source_doc']}" if r["source_doc"] else "",
            f"ref={r['page_or_timestamp']}" if r["page_or_timestamp"] else "",
            f"published={r['date_published']}" if r["date_published"] else "",
            f"unit_as_stated={r['unit_as_stated']}" if r["unit_as_stated"] else "",
            f'quote="{r["quote"]}"' if r["quote"] else "",
            r["note"]] if x)
        rows.append(dict(
            source_id=source_id, entity_type=level, entity_id=entity_id,
            model_id=model_id, provider_id=r["provider_id"] or (cw.provider_of(model_id) if model_id else ""),
            region=r["region"] or "global",
            period_start=r["period_start"] or r["period_end"], period_end=r["period_end"] or r["period_start"],
            variable=r["variable"], value=r["value"], is_estimate=r["is_estimate"] or False,
            source_url=r["source_url"], note=note))
    return make_obs(rows)
