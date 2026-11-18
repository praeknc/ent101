"""Render the source-tier table and variable dictionary to Markdown and CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ._md import md_table

from . import dictionary, sources

TIER_TEXT = {
    "A": "Systematic machine-readable log of behaviour, consistent definition, known (possibly "
         "self-selected) population. Main outcome variables.",
    "B": "Behavioural but indirect proxy, or attributable first-party aggregates that are unaudited "
         "and irregular. Robustness outcomes and level anchors.",
    "C": "Announcements, rankings, surveys, third-party estimates. Description, triangulation, validation.",
}


def render(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = pd.DataFrame([s.to_dict() for s in sources.SOURCES.values()])
    var = pd.DataFrame([v.to_dict() for v in dictionary.VARIABLES.values()])
    src.to_csv(out_dir / "source_registry.csv", index=False)
    var.to_csv(out_dir / "variable_dictionary.csv", index=False)

    md = ["# Demand-side sources: tiers, coverage and sampling bias", "",
          "Generated from `demand_panel/sources.py`. Edit there, then run `python -m demand_panel docs`.", "",
          "## Tier definitions", ""]
    md += [f"- **{t}**: {txt}" for t, txt in TIER_TEXT.items()]
    md += ["", "## Summary", "",
           md_table(src[["source_id", "tier", "category", "entity_level", "native_freq", "region",
                         "collection", "merge_level", "validation_only"]]), ""]
    for s in sources.SOURCES.values():
        md += [f"## `{s.source_id}`: {s.name} (tier {s.tier})", "",
               f"**Can show.** {s.can_show}", "", f"**Cannot show.** {s.cannot_show}", "",
               "**Sampling-bias notes.**", ""] + [f"- {b}" for b in s.sampling_bias] + [""]
        if s.url:
            md += [f"Reference: {s.url}", ""]
    (out_dir / "source_tiers.md").write_text("\n".join(md))

    vd = ["# Variable dictionary (demand-side panel)", "",
          "Generated from `demand_panel/dictionary.py`. `agg` is the rule used to collapse native "
          "frequency to the panel frequency. Derived variables are computed in `panel.py`.", ""]
    for sid, g in var.groupby("source_id", sort=False):
        vd += [f"## {sid} (tier {sources.tier_of(sid)})", "",
               md_table(g[["name", "label", "entity_level", "unit", "native_freq", "agg", "derived",
                          "definition"]]), ""]
    vd += ["## Panel keys and quality columns", "",
           "| column | meaning |", "|---|---|",
           "| model_id | canonical model id (data/reference/model_crosswalk.csv) |",
           "| provider_id | canonical provider id |",
           "| app_id | OpenRouter app slug (app panel) |",
           "| period | pandas Period at panel frequency (W or M) |",
           "| n_obs | raw observations behind the cell (long panel) |",
           "| sources | contributing source_ids |",
           "| tier | best (lowest) tier among contributing sources |",
           "| any_estimate | any contributing value was second-hand, rounded or a lower bound |",
           "| part_a_price_date | date of the Part A price matched as-of period end |",
           "| <price>_lag1 | Part A value in the previous period |", ""]
    (out_dir / "variable_dictionary.md").write_text("\n".join(vd))
    return [out_dir / f for f in ["source_tiers.md", "variable_dictionary.md",
                                  "source_registry.csv", "variable_dictionary.csv"]]
