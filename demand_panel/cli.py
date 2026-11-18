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
    python -m demand_panel merge --prices partA_prices.csv [--index partA_index.csv] [--date-col X --model-col Y]

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


DATE_CANDIDATES = ["date", "effective_date", "price_date", "observed_at", "observation_date",
                   "as_of", "asof", "day", "week", "month", "period", "timestamp", "time"]
MODEL_CANDIDATES = ["model_id", "model", "model_name", "model_slug", "part_a_model_id", "slug"]


def _pick(cols: list[str], explicit: str | None, candidates: list[str], what: str, path) -> str:
    lower = {c.strip().lower().replace(" ", "_"): c for c in cols}
    if explicit:
        if explicit in cols:
            return explicit
        key = explicit.strip().lower().replace(" ", "_")
        if key in lower:
            return lower[key]
        sys.exit(f"{path}: column '{explicit}' not found. Columns are: {cols}")
    for c in candidates:
        if c in lower:
            return lower[c]
    flag = "--date-col" if what == "date" else "--model-col"
    sys.exit(f"{path}: no {what} column found. Columns are: {cols}\n"
             f"Tell the merge which one to use, e.g. {flag} <column name>")


def _load_part_a(path, date_col, model_col, need_model: bool) -> pd.DataFrame:
    """Read a Part A file and rename its date/model columns to 'date'/'model_id'."""
    path = Path(path).expanduser()
    if not path.exists():
        sys.exit(f"File not found: {path}")
    df = pd.read_csv(path)
    cols = list(df.columns)
    ren = {_pick(cols, date_col, DATE_CANDIDATES, "date", path): "date"}
    if need_model:
        ren[_pick(cols, model_col, MODEL_CANDIDATES, "model", path)] = "model_id"
    df = df.rename(columns=ren)
    parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.isna().all():
        sys.exit(f"{path}: could not read any dates in column '{[k for k, v in ren.items() if v == 'date'][0]}'. "
                 f"First values: {df['date'].head(3).tolist()}")
    df["date"] = parsed
    n_bad = int(parsed.isna().sum())
    if n_bad:
        print(f"warning: {n_bad} rows in {path.name} have unreadable dates and were dropped", file=sys.stderr)
        df = df[parsed.notna()]
    num = [c for c in df.columns if c not in ("date", "model_id") and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        sys.exit(f"{path}: no numeric price/index columns found. Columns are: {cols}")
    print(f"{path.name}: using date='{[k for k, v in ren.items() if v == 'date'][0]}'"
          + (f", model='{[k for k, v in ren.items() if v == 'model_id'][0]}'" if need_model else "")
          + f", values={num}", file=sys.stderr)
    return df


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
    m.add_argument("--date-col", help="date column in the prices/index files (auto-detected if omitted)")
    m.add_argument("--model-col", help="model id column in the prices file (auto-detected if omitted)")
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
        from .merge import merge_index, merge_prices
        prices = _load_part_a(args.prices, args.date_col, args.model_col, need_model=True)
        merged, diag = merge_prices(mp, prices, cw, how=args.how)
        if args.index:
            by = [c for c in args.index_by.split(",") if c]
            idx = _load_part_a(args.index, args.date_col, None, need_model=False)
            merged, d2 = merge_index(merged, idx, by=by)
            diag["index"] = d2
        merged.to_csv(out / f"model_panel_merged_{args.freq}.csv", index=False)
        print(json.dumps(diag, indent=2, default=str))


if __name__ == "__main__":
    main()
