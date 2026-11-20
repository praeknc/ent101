"""
Boston Airbnb (Sept 2026) listing-level cleaning and feature engineering.

Implements the rules in Data/Data Cleaning/Boston_Airbnb_Variable_Screening.xlsx.
Category mappings (neighbourhood, property type, room type, amenities) are read
directly from the workbook, so editing the workbook and re-running this script
reproduces the dataset.

Usage (from the project root):
    Project.venv/bin/python Code/clean_listings.py

Outputs:
    Data/Processed/listings_clean.csv            cleaned listing-level dataset
    Data/Processed/data_quality_report.md        sample size, missingness, mappings, anomalies
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_LISTINGS = ROOT / "Data" / "Raw" / "listings.csv"
CONFIG_XLSX = ROOT / "Data" / "Data Cleaning" / "Boston_Airbnb_Variable_Screening.xlsx"
OUT_DIR = ROOT / "Data" / "Processed"
OUT_CSV = OUT_DIR / "listings_clean.csv"
OUT_REPORT = OUT_DIR / "data_quality_report.md"

DAYS_PER_MONTH = 365.25 / 12

REVIEW_SCORE_COLS = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]
MEDIAN_IMPUTE_COLS = (
    ["accommodates", "bedrooms", "beds", "bathrooms"]
    + REVIEW_SCORE_COLS
    + [
        "number_of_reviews",
        "number_of_reviews_ltm",
        "months_since_first_review",
        "days_since_last_review",
        "minimum_nights",
    ]
)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# --------------------------------------------------------------------------- config
def load_config(path: Path) -> dict:
    sheets = pd.read_excel(path, sheet_name=None)

    nbhd = sheets["Neighborhood Mapping"].dropna(subset=["Original Neighborhood"])
    prop = sheets["Property Type Mapping"].dropna(subset=["Original Property Type"])
    room = sheets["Room Type"].dropna(subset=["Room Type"])
    amen = sheets["Amenities Mapping"].dropna(subset=["Engineered Feature", "Root Word / Phrase"])

    return {
        "neighbourhood": dict(zip(nbhd["Original Neighborhood"].str.strip(), nbhd["Proposed Group"].str.strip())),
        "property_type": dict(zip(prop["Original Property Type"].str.strip(), prop["Proposed Group"].str.strip())),
        "room_types": room["Room Type"].str.strip().tolist(),
        "variable_order": (
            sheets["Master Variable List"]["Variable"].dropna().astype(str).str.strip()
            .loc[lambda v: ~v.str.startswith("Total")].tolist()
        ),
        **parse_amenity_rules(amen),
    }


def norm(text: str) -> str:
    """Lowercase and drop whitespace, so 'Dish washer' and 'Dishwasher' compare equal."""
    return re.sub(r"\s+", "", str(text).lower())


def parse_amenity_rules(amen: pd.DataFrame) -> dict:
    """
    Root cells look like 'washer' or 'washer (exclude Dish washer)'. Returns
    feature -> [roots] and feature -> [exclusion phrases], all lowercase.
    Multiple exclusions may be separated by ',' or ';'.
    """
    roots: dict[str, list[str]] = {}
    excludes: dict[str, list[str]] = {}
    for feat, cell in zip(amen["Engineered Feature"].str.strip(), amen["Root Word / Phrase"].astype(str)):
        m = re.match(r"^\s*(.*?)\s*(?:\(\s*exclude\s+(.*?)\s*\))?\s*$", cell, flags=re.I)
        roots.setdefault(feat, []).append(m.group(1).lower())
        excludes.setdefault(feat, [])
        if m.group(2):
            excludes[feat] += [e.strip().lower() for e in re.split(r"[;,]", m.group(2)) if e.strip()]
    return {"amenities": roots, "amenity_excludes": excludes}


# --------------------------------------------------------------------------- helpers
def parse_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"[$,\s]", "", regex=True), errors="coerce")


def parse_bathrooms(text) -> tuple[float, float]:
    """Return (number_of_bathrooms, is_shared) from bathrooms_text."""
    if pd.isna(text):
        return np.nan, 0
    t = str(text).strip().lower()
    shared = 1 if "shared" in t else 0
    m = re.search(r"\d+(\.\d+)?", t)
    if m:
        n = float(m.group())
    elif "half-bath" in t or "half bath" in t:
        n = 0.5
    else:
        n = np.nan
    return n, shared


def parse_amenities(raw) -> list[str]:
    if pd.isna(raw):
        return []
    try:
        return [str(a).strip() for a in json.loads(raw)]
    except (json.JSONDecodeError, TypeError):
        return []


TRACE_COLS = ["id", "host_id", "last_scraped", "price_quote_checkin_date", "price_quote_checkout_date",
              "first_review", "last_review"]


def output_columns(variable: str, out: pd.DataFrame, cfg: dict) -> list[str]:
    """Output columns produced by one Master Variable List row, in workbook order."""
    def dummies(prefix):
        return [c for c in out.columns if c.startswith(prefix + "_") and c != "room_type"]

    special = {
        "price": ["price", "log_price"],
        "neighbourhood_area": ["neighbourhood_cleansed", "neighbourhood_area", *dummies("nbhd")],
        "property_type": ["property_type", "property_group", *dummies("prop")],
        "room_type": ["room_type", *dummies("room")],
        "amenities": list(cfg["amenities"]),
        "host_time": ["total_host_experience_months"],
        "host_listings_count": ["log1p_host_listings_count"],
        "first_review": ["months_since_first_review"],
        "last_review": ["days_since_last_review"],
    }
    return special.get(variable, [variable])


def order_columns(out: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    order = list(TRACE_COLS)
    for v in cfg["variable_order"]:
        order += [c for c in output_columns(v, out, cfg) if c in out.columns and c not in order]
    leftover = [c for c in out.columns if c not in order]
    if leftover:
        print(f"Note: columns not tied to a Master Variable List row, appended at end: {leftover}")
    return out[order + leftover]


def one_hot(series: pd.Series, levels: list[str], prefix: str) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}_{slug(lvl)}": (series == lvl).astype(int) for lvl in levels}, index=series.index)


# --------------------------------------------------------------------------- pipeline
def run() -> None:
    cfg = load_config(CONFIG_XLSX)
    raw = pd.read_csv(RAW_LISTINGS, low_memory=False)
    log: dict = {"n_raw": len(raw), "anomalies": []}

    # ---- 1) sample filter: valid positive price and >= 1 review ----------------
    price = parse_price(raw["price"])
    unparseable = raw["price"].notna() & price.isna()
    keep_price = price.notna() & (price > 0)
    keep_reviews = raw["number_of_reviews"].fillna(0) > 0
    log["filter"] = {
        "price missing": int(raw["price"].isna().sum()),
        "price unparseable": int(unparseable.sum()),
        "price <= 0": int((price <= 0).sum()),
        "valid price": int(keep_price.sum()),
        "valid price & number_of_reviews == 0": int((keep_price & ~keep_reviews).sum()),
    }
    df = raw.loc[keep_price & keep_reviews].copy()
    df["price"] = price[df.index]
    log["n_clean"] = len(df)
    pre_missing = df.isna().sum()

    out = pd.DataFrame(index=df.index)

    # ---- traceability ----------------------------------------------------------
    out["id"] = df["id"]
    out["host_id"] = df["host_id"]
    out["last_scraped"] = pd.to_datetime(df["last_scraped"]).dt.date
    for c in ["price_quote_checkin_date", "price_quote_checkout_date", "first_review", "last_review"]:
        out[c] = pd.to_datetime(df[c], errors="coerce").dt.date

    # ---- target ----------------------------------------------------------------
    out["price"] = df["price"]
    out["log_price"] = np.log(df["price"])

    # ---- location: neighbourhood -> area ---------------------------------------
    nb = df["neighbourhood_cleansed"].str.strip()
    out["neighbourhood_cleansed"] = nb
    out["neighbourhood_area"] = nb.map(cfg["neighbourhood"]).fillna("Unmapped")
    nb_levels = list(dict.fromkeys(cfg["neighbourhood"].values()))

    # ---- property type -> group ------------------------------------------------
    pt = df["property_type"].str.strip()
    out["property_type"] = pt
    out["property_group"] = pt.map(cfg["property_type"]).fillna("Unmapped")
    pt_levels = list(dict.fromkeys(cfg["property_type"].values()))

    # ---- room type -------------------------------------------------------------
    rt = df["room_type"].str.strip()
    out["room_type"] = rt.where(rt.isin(cfg["room_types"]), "Unmapped")

    for col, levels in [("neighbourhood_area", nb_levels), ("property_group", pt_levels),
                        ("room_type", cfg["room_types"])]:
        unmapped = df.loc[out[col] == "Unmapped", {"neighbourhood_area": "neighbourhood_cleansed",
                                                     "property_group": "property_type",
                                                     "room_type": "room_type"}[col]]
        if len(unmapped):
            log["anomalies"].append(
                f"{len(unmapped)} listings with unmapped `{col}` source values: "
                f"{unmapped.value_counts().to_dict()} (one-hot columns all 0 for these rows)."
            )

    # ---- property size ---------------------------------------------------------
    out["accommodates"] = df["accommodates"]
    out["bedrooms"] = df["bedrooms"]
    out["beds"] = df["beds"]
    baths = df["bathrooms_text"].apply(parse_bathrooms)
    out["bathrooms"] = baths.str[0]
    out["bathrooms_share"] = baths.str[1].astype(int)
    unparsed_bath = df["bathrooms_text"].notna() & out["bathrooms"].isna()
    if unparsed_bath.any():
        log["anomalies"].append(
            f"bathrooms_text values not parsable to a number: "
            f"{df.loc[unparsed_bath, 'bathrooms_text'].value_counts().to_dict()}"
        )
    zero_bath = int((out["bathrooms"] == 0).sum())
    if zero_bath:
        log["anomalies"].append(
            f"{zero_bath} listings report 0 bathrooms ('0 baths' / '0 shared baths'); kept as 0, not imputed."
        )

    # ---- amenities -------------------------------------------------------------
    amen_lists = df["amenities"].apply(parse_amenities)
    amen_lower = amen_lists.apply(lambda xs: [a.lower() for a in xs])
    log["amenity_matches"] = {}
    log["amenity_excluded"] = {}
    for feat, roots in cfg["amenities"].items():
        excl = [norm(e) for e in cfg["amenity_excludes"][feat]]

        def matches(label: str, r: str) -> bool:
            return r in label.lower() and not any(e in norm(label) for e in excl)

        out[feat] = amen_lists.apply(lambda xs: int(any(matches(a, r) for a in xs for r in roots)))
        log["amenity_matches"][feat] = {
            r: pd.Series([a for xs in amen_lists for a in xs if matches(a, r)]).value_counts() for r in roots
        }
        if excl:
            log["amenity_excluded"][feat] = pd.Series(
                [a for xs in amen_lists for a in xs
                 if any(r in a.lower() for r in roots) and any(e in norm(a) for e in excl)]
            ).value_counts()

    # known substring false positives the literal rule still picks up
    def only_matched_by(labels_fp: set[str], roots: list[str]) -> int:
        def f(xs):
            hits = {a for a in xs if any(r in a for r in roots)}
            return bool(hits) and hits <= labels_fp
        return int(amen_lower.apply(f).sum())

    n = only_matched_by({"garden view"}, cfg["amenities"]["has_outdoor_space"])
    if n:
        log["anomalies"].append(
            f"`has_outdoor_space` = 1 for {n} listings solely because of 'Garden view' (substring overlap "
            "allowed by the workbook's matching rule). Consider tightening the roots if unintended."
        )

    # ---- host ------------------------------------------------------------------
    out["total_host_experience_months"] = 12 * df["hosts_time_as_host_years"] + df["hosts_time_as_host_months"]
    for c in ["host_is_superhost", "host_identity_verified"]:
        v = df[c].astype("string").str.strip().str.lower()
        bad = v.notna() & ~v.isin(["t", "f"])
        if bad.any():
            log["anomalies"].append(f"`{c}` non t/f values: {v[bad].value_counts().to_dict()} (set to 0).")
        out[c] = (v == "t").astype(int)
    out["log1p_host_listings_count"] = np.log1p(df["host_listings_count"])

    # ---- reviews ---------------------------------------------------------------
    for c in REVIEW_SCORE_COLS + ["number_of_reviews", "number_of_reviews_ltm"]:
        out[c] = df[c]
    ref = pd.to_datetime(df["last_scraped"])
    fr = pd.to_datetime(df["first_review"], errors="coerce")
    lr = pd.to_datetime(df["last_review"], errors="coerce")
    out["months_since_first_review"] = (ref - fr).dt.days / DAYS_PER_MONTH
    out["days_since_last_review"] = (ref - lr).dt.days
    if ((ref - fr).dt.days < 0).any() or ((ref - lr).dt.days < 0).any():
        log["anomalies"].append("Some review dates fall after last_scraped.")
    if (lr < fr).any():
        log["anomalies"].append(f"{int((lr < fr).sum())} listings with last_review before first_review.")

    # ---- booking rules ---------------------------------------------------------
    out["minimum_nights"] = df["minimum_nights"]
    ci = pd.to_datetime(df["price_quote_checkin_date"], errors="coerce")
    co = pd.to_datetime(df["price_quote_checkout_date"], errors="coerce")
    out["stay_length"] = (co - ci).dt.days
    if (out["stay_length"] <= 0).any():
        log["anomalies"].append(f"{int((out['stay_length'] <= 0).sum())} non-positive stay_length values.")
    if out["stay_length"].isna().any():
        log["anomalies"].append(f"{int(out['stay_length'].isna().sum())} listings with missing quote dates "
                                "(stay_length left missing).")
    short = (out["stay_length"] < df["minimum_nights"]).sum()
    if short:
        log["anomalies"].append(f"{int(short)} quotes shorter than minimum_nights.")
    early = (ci < ref).sum()
    if early:
        log["anomalies"].append(f"{int(early)} quote check-in dates before last_scraped.")
    if "price_quote_price_per_night" in df:
        diff = (df["price"] - df["price_quote_price_per_night"]).abs()
        log["price_vs_quote"] = (int((diff > 0.01).sum()), int(df["price_quote_price_per_night"].isna().sum()))

    # ---- median imputation (medians from the filtered sample) ------------------
    log["impute"] = {}
    for c in MEDIAN_IMPUTE_COLS:
        n_miss = int(out[c].isna().sum())
        med = float(out[c].median())
        log["impute"][c] = (n_miss, med)
        out[c] = out[c].fillna(med)

    # ---- one-hot encodings (full; drop a reference level for linear models) ----
    out = pd.concat(
        [
            out,
            one_hot(out["neighbourhood_area"], nb_levels, "nbhd"),
            one_hot(out["property_group"], pt_levels, "prop"),
            one_hot(out["room_type"], cfg["room_types"], "room"),
        ],
        axis=1,
    )

    # ---- outlier notes (kept, not removed) -------------------------------------
    q1, q3 = out["log_price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = int(((out["log_price"] < q1 - 3 * iqr) | (out["log_price"] > q3 + 3 * iqr)).sum())
    log["outliers"] = {
        "price_max": float(out["price"].max()),
        "price_min": float(out["price"].min()),
        "log_price_3iqr": n_out,
        "accommodates_max": int(out["accommodates"].max()),
        "bedrooms_max": int(out["bedrooms"].max()),
        "beds_max": float(out["beds"].max()),
        "bathrooms_max": float(out["bathrooms"].max()),
        "minimum_nights_max": float(out["minimum_nights"].max()),
        "beds_gt_3x_accommodates": int((out["beds"] > 3 * out["accommodates"]).sum()),
    }

    out = order_columns(out, cfg)
    assert out["id"].is_unique, "duplicate listing ids"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.reset_index(drop=True).to_csv(OUT_CSV, index=False)
    write_report(out, df, cfg, log, pre_missing, nb_levels, pt_levels)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}  shape={out.shape}")
    print(f"Wrote {OUT_REPORT.relative_to(ROOT)}")


# --------------------------------------------------------------------------- report
def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(out, df, cfg, log, pre_missing, nb_levels, pt_levels) -> None:
    n = log["n_clean"]
    L = [
        "# Data Quality Report: Boston Airbnb listings (Sept 2026)",
        "",
        f"Generated by `Code/clean_listings.py` from `{RAW_LISTINGS.relative_to(ROOT)}` using rules in "
        f"`{CONFIG_XLSX.relative_to(ROOT)}`.",
        "",
        "## 1. Sample size",
        "",
        f"- Raw listings: **{log['n_raw']:,}**",
    ]
    L += [f"- {k}: {v:,}" for k, v in log["filter"].items()]
    L += [
        f"- **Retained (valid price > 0 and number_of_reviews > 0): {n:,}**",
        f"- Output columns: {out.shape[1]}; listing `id` unique: {out['id'].is_unique}",
        "",
        "## 2. Missing values in retained sample (before imputation)",
        "",
        "Source columns used by the pipeline:",
        "",
    ]
    used = ["price", "neighbourhood_cleansed", "property_type", "room_type", "accommodates", "bedrooms", "beds",
            "bathrooms_text", "amenities", "hosts_time_as_host_years", "hosts_time_as_host_months",
            "host_is_superhost", "host_identity_verified", "host_listings_count", *REVIEW_SCORE_COLS,
            "number_of_reviews", "number_of_reviews_ltm", "first_review", "last_review", "minimum_nights",
            "price_quote_checkin_date", "price_quote_checkout_date"]
    miss = pd.DataFrame({"column": used,
                         "missing_n": [int(pre_missing[c]) for c in used],
                         "missing_%": [round(100 * pre_missing[c] / n, 2) for c in used]})
    L += [md_table(miss), "", "Median imputation applied (medians from the retained sample):", ""]
    imp = pd.DataFrame([(c, m, round(v, 3)) for c, (m, v) in log["impute"].items()],
                       columns=["feature", "imputed_n", "median"])
    L += [md_table(imp), "",
          "Other rules: `bathrooms_share` = 0 when not inferable; `host_is_superhost` / "
          "`host_identity_verified` = 0 when missing.", "",
          "## 3. Category mappings", "", "### Neighbourhood → neighbourhood_area", ""]

    def mapping_table(src, grp):
        t = (out.groupby([grp, src]).size().rename("n").reset_index()
             .assign(**{"%": lambda x: (100 * x["n"] / n).round(2)}))
        t[grp] = pd.Categorical(t[grp], categories=list(dict.fromkeys(list(t[grp].unique()))), ordered=True)
        return t.sort_values([grp, "n"], ascending=[True, False])

    def group_totals(grp, levels):
        vc = out[grp].value_counts()
        return pd.DataFrame({"group": levels, "n": [int(vc.get(l, 0)) for l in levels],
                             "%": [round(100 * vc.get(l, 0) / n, 2) for l in levels]})

    L += [md_table(group_totals("neighbourhood_area", nb_levels)), "",
          "<details><summary>Original → group detail</summary>", "",
          md_table(mapping_table("neighbourhood_cleansed", "neighbourhood_area")), "", "</details>", "",
          "### Property type → property_group", "",
          md_table(group_totals("property_group", pt_levels)), "",
          "<details><summary>Original → group detail</summary>", "",
          md_table(mapping_table("property_type", "property_group")), "", "</details>", "",
          "### Room type", "", md_table(group_totals("room_type", cfg["room_types"])), "",
          "Rare levels (< 1% of sample): " + ", ".join(
              f"`{c}` ({int(out[c].sum())})" for c in out.columns
              if c.startswith(("nbhd_", "prop_", "room_")) and c != "room_type" and out[c].mean() < 0.01) + ".",
          "", "### Bathrooms parsing", "",
          md_table(df.assign(bathrooms=out["bathrooms"], bathrooms_share=out["bathrooms_share"])
                   .groupby(df["bathrooms_text"].fillna("<missing>"))
                   .agg(n=("id", "size"), bathrooms=("bathrooms", "first"), bathrooms_share=("bathrooms_share", "first"))
                   .reset_index().sort_values("n", ascending=False)),
          "", "### Amenities (case-insensitive substring match)", ""]
    amen_rows = []
    for feat, per_root in log["amenity_matches"].items():
        amen_rows.append((feat, ", ".join(f"`{r}`" for r in per_root), int(out[feat].sum()),
                          round(100 * out[feat].mean(), 1)))
    L += [md_table(pd.DataFrame(amen_rows, columns=["feature", "roots", "listings = 1", "%"])), ""]
    L += ["<details><summary>Raw amenity labels matched per root (top 10 by occurrences)</summary>", ""]
    for feat, per_root in log["amenity_matches"].items():
        L.append(f"**{feat}**")
        L.append("")
        for r, vc in per_root.items():
            top = "; ".join(f"{k} ({v})" for k, v in vc.head(10).items()) or "_no matches_"
            L.append(f"- `{r}` → {len(vc)} distinct labels: {top}")
        L.append("")
    L += ["</details>", ""]
    for feat, vc in log["amenity_excluded"].items():
        L += [f"Excluded from `{feat}` (workbook exclude rules): "
              + "; ".join(f"{k} ({v})" for k, v in vc.items()), ""]
    L += ["## 4. Unexpected values and checks", ""]
    L += [f"- {a}" for a in log["anomalies"]] or ["- None."]
    if "price_vs_quote" in log:
        d, m = log["price_vs_quote"]
        L.append(f"- `price` differs from `price_quote_price_per_night` for {d} listings "
                 f"({m} have no quote price-per-night).")
    o = log["outliers"]
    L += [
        f"- Price range ${o['price_min']:,.2f} – ${o['price_max']:,.2f}; {o['log_price_3iqr']} listings outside "
        "3×IQR on log_price. **Kept** (no outlier removal specified in workbook).",
        f"- Max accommodates {o['accommodates_max']}, bedrooms {o['bedrooms_max']}, beds {o['beds_max']:.0f}, "
        f"bathrooms {o['bathrooms_max']:.1f}, minimum_nights {o['minimum_nights_max']:.0f}.",
        f"- Listings with beds > 3 × accommodates: {o['beds_gt_3x_accommodates']}.",
        f"- stay_length distribution (nights): "
        + ", ".join(f"{int(k)}: {v}" for k, v in out["stay_length"].value_counts().head(8).items()) + " …",
        "",
        "## 5. Notes for modelling",
        "",
        "- Medians are computed on the full retained sample. For strict out-of-sample evaluation, refit "
        "imputation medians on the training fold only.",
        "- One-hot encodings are full (all levels). Drop one reference level per block for "
        "intercept-based linear models.",
        "- `price`, `id`, `host_id`, and date columns are kept for traceability and should be excluded "
        "from the feature matrix (use `log_price` as the target).",
        "",
    ]
    OUT_REPORT.write_text("\n".join(L))


if __name__ == "__main__":
    run()
