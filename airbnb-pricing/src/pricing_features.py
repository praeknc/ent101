"""Data pre-check, cleaning and feature engineering shared by both notebooks.

Every rule here was decided from the checks in notebooks/boston_airbnb_pricing.ipynb
(section 2 "Data pre-check" and section 4 "Variable screening"). The browser
calculator (interface/pricing.js) implements the same build_features() logic.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CENTER = (42.3550, -71.0656)          # Boston Common
PRICE_RANGE = (30, 2000)              # nightly USD kept for modelling
RPM_CAP = 15.0                        # reviews per month above this are capped
HOST_BANDS = [0, 1, 3, 10, 50, np.inf]  # 1 | 2-3 | 4-10 | 11-50 | 51+ listings -> band 0..4
ROOM_MERGE = {"Shared room": "Private room"}          # 4 listings: too few to learn from
HOOD_MERGE = {"Leather District": "Chinatown"}         # 3 listings; adjacent, similar prices
HOOD_DROP = {"Harbor Islands"}                         # no listings at all

# Title keywords kept after screening ("managed brand" was dropped: no signal)
KEYWORDS = {
    "kw_parking":   r"parking|garage",
    "kw_luxury":    r"luxur|modern|renovat|designer|stylish|chic|upscale|elegant",
    "kw_view":      r"view|rooftop|roof deck|skyline|waterfront|harbor",
    "kw_amenities": r"gym|pool|fitness|concierge|doorman",
}


def model_neighbourhoods(official):
    """Neighbourhood categories the model uses (official list minus merged/empty ones)."""
    return sorted(set(official) - set(HOOD_MERGE) - HOOD_DROP)


def _in_polygons(geo):
    from matplotlib.path import Path as MPath
    polys = {f["properties"]["neighbourhood"]: [MPath(np.array(p[0])) for p in f["geometry"]["coordinates"]]
             for f in geo["features"]}
    return polys


def precheck(df, official_hoods, geo, require_price=True):
    """Run the data-quality checks. Returns (clean dataframe, report dataframe).

    Each report row: check, rows affected, action taken. With require_price=False
    (snapshots that publish no prices) the price rules are skipped and every other
    rule still applies.
    """
    d = df.copy()
    rows = []

    def log(check, n, action):
        rows.append({"check": check, "rows": int(n), "action": action})

    log("Duplicate listing IDs", d["id"].duplicated().sum(), "drop repeats")
    d = d[~d["id"].duplicated()]

    if require_price:
        n = d["price"].isna().sum()
        log("Missing price (inactive listing)", n, "drop: no target")
        d = d[d["price"].notna()]

        lo, hi = PRICE_RANGE
        n = (~d["price"].between(lo, hi)).sum()
        log(f"Price outside ${lo}-${hi}", n, "drop: extreme or erroneous price")
        d = d[d["price"].between(lo, hi)]

    key = ["host_id", "name", "room_type", "latitude", "longitude", "price", "minimum_nights"]
    # only priced rows can be compared on price; two blank prices are not evidence of a copy
    dup = d.duplicated(subset=key) & d["price"].notna()
    log("Exact duplicate posting (same host, title, location, price, rules)", dup.sum(), "drop copies")
    d = d[~dup]

    n = (~d["neighbourhood"].isin(official_hoods)).sum()
    log("Neighbourhood not in official list", n, "drop")
    d = d[d["neighbourhood"].isin(official_hoods)]

    polys = _in_polygons(geo)
    inside = np.array([any(p.contains_point((x, y)) for p in polys[h])
                       for x, y, h in zip(d["longitude"], d["latitude"], d["neighbourhood"])])
    log("Coordinates outside their neighbourhood boundary", (~inside).sum(), "flag (kept)")

    n = (d["number_of_reviews_ltm"] > d["number_of_reviews"]).sum()
    log("Reviews in last 12 months > total reviews", n, "drop: inconsistent")
    d = d[~(d["number_of_reviews_ltm"] > d["number_of_reviews"])]

    n = d["reviews_per_month"].isna().sum()
    log("Reviews per month missing (no reviews yet)", n, "fill with 0")
    d["reviews_per_month"] = d["reviews_per_month"].fillna(0)

    n = ((d["number_of_reviews"] == 0) & (d["reviews_per_month"] > 0)).sum()
    log("Reviews per month > 0 with zero reviews", n, "set to 0")
    d.loc[d["number_of_reviews"] == 0, "reviews_per_month"] = 0

    n = (d["reviews_per_month"] > RPM_CAP).sum()
    log(f"Reviews per month above {RPM_CAP:g}", n, f"cap at {RPM_CAP:g}")
    d["reviews_per_month"] = d["reviews_per_month"].clip(upper=RPM_CAP)

    med = d["minimum_nights"].median()
    n = d["minimum_nights"].isna().sum()
    log("Minimum nights missing", n, f"fill with median ({med:g})")
    d["minimum_nights"] = d["minimum_nights"].fillna(med)

    n = (d["minimum_nights"] > 365).sum()
    log("Minimum nights above 365", n, "cap at 365")
    d["minimum_nights"] = d["minimum_nights"].clip(upper=365)

    n = d["calculated_host_listings_count"].isna().sum()
    log("Host listing count missing", n, "fill with 1")
    d["calculated_host_listings_count"] = d["calculated_host_listings_count"].fillna(1)

    n = (d["availability_365"] == 0).sum()
    log("Calendar closed for the next year (availability 0)", n,
        "kept: these listings are actively reviewed, i.e. fully booked")

    for old, new in ROOM_MERGE.items():
        n = (d["room_type"] == old).sum()
        log(f"Rare room type: {old}", n, f"merge into {new}")
        d.loc[d["room_type"] == old, "room_type"] = new
    for old, new in HOOD_MERGE.items():
        n = (d["neighbourhood"] == old).sum()
        log(f"Rare neighbourhood: {old}", n, f"merge into {new}")
        d.loc[d["neighbourhood"] == old, "neighbourhood"] = new

    return d, pd.DataFrame(rows)


def extract_bedrooms(title):
    t = str(title).lower()
    if "studio" in t:
        return 0
    m = re.search(r"(\d+)\s*-?\s*(?:br\b|bd\b|bdr|bed|bedroom|b/r)", t)
    return min(int(m.group(1)), 8) if m else np.nan


def host_band(count):
    """1 | 2-3 | 4-10 | 11-50 | 51+ listings -> 0..4"""
    return pd.cut(pd.Series(count, dtype=float).fillna(1), HOST_BANDS, labels=False).astype(int).values


def build_features(d, neighbourhoods, screened=True):
    """Feature matrix. screened=False reproduces the original (pre-screening) feature set."""
    title = d["name"].fillna("").str.lower()
    beds = d["name"].fillna("").map(extract_bedrooms)
    lic = d["license"].fillna("").str.lower()
    mn = d["minimum_nights"].fillna(2)
    X = pd.DataFrame(index=d.index)
    X["bedrooms"] = beds.fillna(1)
    X["bedrooms_known"] = beds.notna().astype(int)
    X["private_room"] = (d["room_type"] == "Private room").astype(int)
    X["hotel_room"] = (d["room_type"] == "Hotel room").astype(int)
    if not screened:
        X["shared_room"] = (d["room_type"] == "Shared room").astype(int)
    X["log_min_nights"] = np.log1p(mn)
    if not screened:
        X["monthly"] = (mn >= 28).astype(int)
    X["log_reviews"] = np.log1p(d["number_of_reviews"])
    X["reviews_per_month"] = d["reviews_per_month"].fillna(0)
    X["number_of_reviews_ltm"] = d["number_of_reviews_ltm"]
    if screened:
        X["host_size_band"] = host_band(d["calculated_host_listings_count"])
    else:
        X["log_host_listings"] = np.log1p(d["calculated_host_listings_count"].fillna(1))
    X["availability_365"] = d["availability_365"]
    X["licensed"] = lic.str.startswith("str").astype(int)
    X["exempt"] = lic.str.contains("exempt").astype(int)
    X["dist_km"] = np.sqrt(((d["latitude"] - CENTER[0]) * 111.0) ** 2 + ((d["longitude"] - CENTER[1]) * 82.0) ** 2)
    X["latitude"] = d["latitude"]
    X["longitude"] = d["longitude"]
    kws = dict(KEYWORDS)
    if not screened:
        kws["kw_managed"] = r"blueground|sonder|landing|kasa|furnished|corporate"
    for k, pat in kws.items():
        X[k] = title.str.contains(pat).astype(int)
    for h in neighbourhoods:
        X["nb_" + h] = (d["neighbourhood"] == h).astype(int)
    return X.astype(float)


def vif(X):
    """Variance inflation factor of each column (regress it on all the others)."""
    M = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    out = {}
    for c in M.columns:
        others = np.c_[np.ones(len(M)), M.drop(columns=c).values]
        beta = np.linalg.lstsq(others, M[c].values, rcond=None)[0]
        r2 = 1 - (M[c].values - others @ beta).var() / max(M[c].var(ddof=0), 1e-12)
        out[c] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out)


def load_geo(path):
    return json.loads(Path(path).read_text())
