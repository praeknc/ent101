"""Command line entry point.

    python -m demand_panel docs
    python -m demand_panel templates
    python -m demand_panel collect huggingface
    python -m demand_panel collect github --start 2023-01 --end 2026-10
    python -m demand_panel collect pypi
    python -m demand_panel collect npm --start 2024-06-01 --end 2026-10-31
    python -m demand_panel load openrouter data/raw/openrouter_export.csv
    python -m demand_panel load manual
    python -m demand_panel build --freq M --start 2023-01 --end 2026-10
    python -m demand_panel merge --prices partA_prices.csv [--index partA_index.csv]

All raw observations go to data/processed/observations.csv (append, de-duplicated).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import coverage, docs, panel, schema
from .crosswalk import Crosswalk
from .collectors import github, huggingface, manual, openrouter, packages

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OBS = DATA / "processed" / "observations.csv"
UNMATCHED = DATA / "processed" / "unmatched_ids.txt"


def _store(df: pd.DataFrame, unmatched: list[str] | None = None) -> None:
    all_obs = schema.append_obs(df, OBS)
    print(f"+{len(df)} rows -> {OBS} ({len(all_obs)} total)")
    if unmatched:
        prev = set(UNMATCHED.read_text().split()) if UNMATCHED.exists() else set()
        UNMATCHED.write_text("\n".join(sorted(prev | set(unmatched))) + "\n")
        print(f"{len(unmatched)} ids not in crosswalk -> {UNMATCHED}", file=sys.stderr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="demand_panel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("docs")
    sub.add_parser("templates")
    c = sub.add_parser("collect")
    c.add_argument("source", choices=["huggingface", "github", "pypi", "npm", "openrouter-models"])
    c.add_argument("--start", default="2023-01")
    c.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m"))
    c.add_argument("--no-derivatives", action="store_true")
    ld = sub.add_parser("load")
    ld.add_argument("source", choices=["openrouter", "manual", "pypi-bigquery"])
    ld.add_argument("path", nargs="?")
    ld.add_argument("--freq", default="W", help="native freq of an OpenRouter export (W or M)")
    b = sub.add_parser("build")
    b.add_argument("--freq", default="M", choices=["W", "M", "Q"])
    b.add_argument("--start", default="2023-01")
    b.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m"))
    b.add_argument("--include-validation", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--prices", required=True)
    m.add_argument("--index")
    m.add_argument("--index-by", default="", help="comma-separated key columns of the index file")
    m.add_argument("--how", default="asof", choices=["asof", "period_mean"])
    m.add_argument("--freq", default="M")
    args = ap.parse_args(argv)

    cw = Crosswalk.load()
    if args.cmd == "docs":
        for p in docs.render(ROOT / "docs"):
            print("wrote", p)
    elif args.cmd == "templates":
        manual.write_templates(DATA / "manual")
        print("templates in", DATA / "manual")
    elif args.cmd == "collect":
        if args.source == "huggingface":
            _store(huggingface.collect(cw, derivatives=not args.no_derivatives))
        elif args.source == "github":
            q = pd.read_csv(DATA / "reference" / "github_queries.csv", dtype=str)
            _store(github.collect(q, args.start, args.end))
        elif args.source == "pypi":
            _store(packages.collect_pypi(cw))
        elif args.source == "npm":
            _store(packages.collect_npm(cw, args.start, args.end))
        elif args.source == "openrouter-models":
            out = DATA / "raw" / f"openrouter_models_{pd.Timestamp.today():%Y%m%d}.csv"
            openrouter.fetch_models().to_csv(out, index=False)
            print("wrote", out)
    elif args.cmd == "load":
        if args.source == "openrouter":
            df, um = openrouter.load_export(args.path, cw, freq=args.freq)
            _store(df, um)
        elif args.source == "pypi-bigquery":
            _store(packages.load_pypi_bigquery(args.path, cw))
        else:
            for sid, fname in manual.FILES.items():
                p = DATA / "manual" / fname
                if p.exists():
                    _store(manual.load(p, cw, sid))
    elif args.cmd == "build":
        obs = schema.read_obs(OBS)
        out = DATA / "processed"
        mw, ml = panel.build_model_panel(obs, args.freq, args.include_validation)
        pw, pl = panel.build_provider_panel(obs, args.freq, args.include_validation)
        ap_ = panel.build_app_panel(obs, args.freq)
        for name, df in [("model_panel", mw), ("model_panel_long", ml), ("provider_panel", pw),
                         ("provider_panel_long", pl), ("app_panel", ap_)]:
            df.to_csv(out / f"{name}_{args.freq}.csv", index=False)
            print(f"wrote {name}_{args.freq}.csv ({len(df)} rows)")
        um = UNMATCHED.read_text().split() if UNMATCHED.exists() else None
        coverage.write_report(obs, cw, ROOT / "docs" / "coverage_report.md", args.start, args.end,
                              args.freq, um)
        print("wrote docs/coverage_report.md")
    elif args.cmd == "merge":
        out = DATA / "processed"
        mp = pd.read_csv(out / f"model_panel_{args.freq}.csv")
        mp["period"] = pd.PeriodIndex(mp["period"], freq=args.freq)
        from .merge import merge_prices
        merged, diag = merge_prices(
            mp, pd.read_csv(args.prices), cw, how=args.how)
        if args.index:
            from .merge import merge_index
            by = [c for c in args.index_by.split(",") if c]
            merged, d2 = merge_index(merged, pd.read_csv(args.index), by=by)
            diag["index"] = d2
        merged.to_csv(out / f"model_panel_merged_{args.freq}.csv", index=False)
        print(json.dumps(diag, indent=2, default=str))


if __name__ == "__main__":
    main()
