"""Phase 3 features: detailed listing attributes (Step 1) and location context (Step 2).

Used by notebooks/boston_airbnb_phase3_enriched.ipynb. Builds on pricing_features.py:
the same pre-check runs first, then these extra columns are added.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import pricing_features as pf

# ---------------------------------------------------------------- Step 1: detailed attributes

# Columns that must never be features: they contain the price or are outcomes of it.
LEAKY = {
    "price_quote_checkin_date", "price_quote_checkout_date", "price_quote_total_price",
    "price_quote_price_per_night", "price_quote_raw",           # the price itself
    "estimated_revenue_l365d",                                  # price x occupancy
    "estimated_occupancy_l365d", "availability_30", "availability_60", "availability_90",
    "availability_eoy",                                         # bookings, i.e. outcomes of the price
}

# Amenity groups: regex over the lower-cased amenity strings. Groups are stable across
# quarters, unlike raw strings such as "65 inch HDTV" vs "TV with standard cable".
AMENITY_GROUPS = {
    "am_free_parking":   r"free (?:parking|street parking|driveway|residential garage)",
    "am_paid_parking":   r"paid parking",
    "am_washer_in_unit": r"washer.*in unit|in-unit washer|^washer$",
    "am_dryer":          r"dryer",
    "am_dishwasher":     r"dishwasher",
    "am_central_ac":     r"central air",
    "am_any_ac":         r"air conditioning|ac - split",
    "am_gym":            r"\bgym\b|fitness",
    "am_pool":           r"\bpool\b",
    "am_hot_tub":        r"hot tub",
    "am_elevator":       r"elevator",
    "am_workspace":      r"workspace",
    "am_self_checkin":   r"self check-in|smart lock|keypad|lockbox",
    "am_balcony_patio":  r"balcony|patio",
    "am_view":           r"(?:city skyline|water|harbor|bay|ocean|park|courtyard) view",
    "am_kitchen":        r"^kitchen$",
    "am_bathtub":        r"bathtub",
    "am_fireplace":      r"fireplace",
    "am_ev_charger":     r"ev charger",
    "am_pets":           r"pets allowed",
    "am_long_term":      r"long term stays",
    "am_luggage":        r"luggage",
    "am_crib":           r"crib",
    "am_private_entry":  r"private entrance",
    "am_backyard":       r"backyard",
    "am_bbq":            r"bbq|grill",
    "am_tv":             r"\btv\b|hdtv",
    "am_streaming":      r"netflix|hulu|disney|amazon prime|apple tv|chromecast|roku|hbo",
    "am_game_console":   r"game console|ps4|ps5|xbox|nintendo",
    "am_sound_system":   r"sound system|bluetooth",
}

PROPERTY_GROUPS = {
    "Entire rental unit": "rental unit", "Entire condo": "condo", "Entire home": "house",
    "Entire townhouse": "house", "Entire serviced apartment": "serviced apt", "Entire guest suite": "guest suite",
    "Entire loft": "loft", "Room in hotel": "hotel", "Room in boutique hotel": "hotel", "Room in aparthotel": "hotel",
    "Private room in rental unit": "room in unit", "Private room in condo": "room in unit",
    "Private room in home": "room in house", "Private room in townhouse": "room in house",
    "Private room in bed and breakfast": "b&b", "Room in bed and breakfast": "b&b",
}
REVIEW_SCORES = ["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
                 "review_scores_checkin", "review_scores_communication", "review_scores_location",
                 "review_scores_value"]


def load_detailed(path):
    """Read a detailed Inside Airbnb export and map it onto the summary-file column names
    that pricing_features.precheck() expects."""
    d = pd.read_csv(path, low_memory=False)
    d["price"] = pd.to_numeric(d["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    d["neighbourhood"] = d["neighbourhood_cleansed"]
    return d


def parse_baths(text):
    """'1.5 shared baths' -> (1.5, shared=1); 'Half-bath' -> 0.5; missing -> (nan, 0)."""
    t = str(text).lower()
    if t in ("nan", ""):
        return np.nan, 0
    shared = int("shared" in t)
    if "half" in t:
        return 0.5, shared
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    return (float(m.group(1)) if m else np.nan), shared


def amenity_flags(amenities):
    """JSON-ish list string -> dict of group flags plus a count."""
    items = [a.strip().strip('"').lower() for a in str(amenities).strip("[]").split('", "')]
    items = [a for a in items if a]
    out = {g: int(any(re.search(p, a) for a in items)) for g, p in AMENITY_GROUPS.items()}
    out["am_count"] = len(items)
    return out


def detailed_features(d, property_levels):
    """Step 1 features. NaN is kept where a value is genuinely unknown (tree models handle it);
    linear models get a median fill plus a 'missing' flag elsewhere."""
    X = pd.DataFrame(index=d.index)
    X["accommodates"] = d["accommodates"].astype(float)
    beds_title = d["name"].map(pf.extract_bedrooms)
    X["bedrooms_d"] = d["bedrooms"].astype(float).fillna(beds_title)   # detailed first, title as fallback
    X["beds"] = d["beds"].astype(float)
    b = d["bathrooms_text"].map(parse_baths)
    X["baths"] = [v for v, _ in b]
    X["baths_shared"] = [s for _, s in b]
    prop = d["property_type"].map(PROPERTY_GROUPS).fillna("other")
    for lvl in property_levels:
        X["prop_" + lvl.replace(" ", "_").replace("&", "and")] = (prop == lvl).astype(int)
    am = pd.DataFrame([amenity_flags(a) for a in d["amenities"]], index=d.index)
    X = pd.concat([X, am], axis=1)
    for c in REVIEW_SCORES:
        X[c.replace("review_scores_", "rs_")] = d[c].astype(float)
    X["has_review_scores"] = d["review_scores_rating"].notna().astype(int)
    X["superhost"] = (d["host_is_superhost"] == "t").astype(int)
    X["identity_verified"] = (d["host_identity_verified"] == "t").astype(int)
    X["desc_len"] = d["description"].fillna("").str.len()
    return X


# ---------------------------------------------------------------- Step 2: location context

# Approximate coordinates of well-known Boston demand drivers (accurate to roughly 100-200 m).
LANDMARKS = {
    "boston_common":    (42.3550, -71.0656),
    "logan_airport":    (42.3656, -71.0096),
    "fenway_park":      (42.3467, -71.0972),
    "convention_ctr":   (42.3456, -71.0451),
    "seaport":          (42.3529, -71.0430),
    "south_station":    (42.3523, -71.0552),
    "td_garden":        (42.3662, -71.0621),
    "back_bay_station": (42.3474, -71.0754),
    "faneuil_hall":     (42.3600, -71.0560),
    "mit":              (42.3601, -71.0942),
    "harvard_sq":       (42.3732, -71.1189),
    "longwood_medical": (42.3376, -71.1060),
    "northeastern":     (42.3398, -71.0892),
    "boston_univ":      (42.3505, -71.1054),
}


def km_coords(lat, lon):
    return np.c_[(np.asarray(lat) - pf.CENTER[0]) * 111.0, (np.asarray(lon) - pf.CENTER[1]) * 82.0]


def landmark_features(d):
    X = pd.DataFrame(index=d.index)
    P = km_coords(d["latitude"], d["longitude"])
    dists = []
    for name, (la, lo) in LANDMARKS.items():
        q = km_coords([la], [lo])[0]
        dist = np.sqrt(((P - q) ** 2).sum(1))
        X["d_" + name] = dist
        dists.append(dist)
    D = np.vstack(dists)
    X["d_nearest_landmark"] = D.min(0)
    X["n_landmarks_2km"] = (D < 2).sum(0)
    return X


def density_features(d, supply):
    """Listings (priced or not) of the same quarter within 300 m and 1 km: competition / popularity."""
    X = pd.DataFrame(index=d.index)
    for q in d["quarter"].unique():
        m = (d["quarter"] == q).values
        A = km_coords(d.loc[m, "latitude"], d.loc[m, "longitude"])
        S = supply[supply["quarter"] == q]
        B = km_coords(S["latitude"], S["longitude"])
        dist = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        X.loc[m, "supply_300m"] = (dist < 0.3).sum(1) - 1
        X.loc[m, "supply_1km"] = (dist < 1.0).sum(1) - 1
    return X


def knn_market_price(query, ref, k=15):
    """Local Airbnb market level: median log price of the k nearest *priced* listings of OTHER hosts
    in the same quarter, drawn only from `ref` (the training rows). Also returns the same median
    restricted to the query's room type, which compares like with like."""
    out = pd.DataFrame(index=query.index, columns=["mkt_logprice_k15", "mkt_logprice_same_room"], dtype=float)
    for q in query["quarter"].unique():
        qm = query["quarter"] == q
        R = ref[ref["quarter"] == q]
        Rk, Ry, Rh, Rr = km_coords(R["latitude"], R["longitude"]), np.log(R["price"].values), R["host_id"].values, R["room_type"].values
        Q = query[qm]
        Qk = km_coords(Q["latitude"], Q["longitude"])
        dist = np.sqrt(((Qk[:, None, :] - Rk[None, :, :]) ** 2).sum(-1))
        dist[Q["host_id"].values[:, None] == Rh[None, :]] = np.inf          # never use the same host
        order = np.argsort(dist, axis=1)
        all_k = np.array([np.median(Ry[o[:k]]) for o in order])
        same = []
        for i, o in enumerate(order):
            o2 = o[Rr[o] == Q["room_type"].values[i]][:k]
            same.append(np.median(Ry[o2]) if len(o2) else all_k[i])
        out.loc[Q.index, "mkt_logprice_k15"] = all_k
        out.loc[Q.index, "mkt_logprice_same_room"] = same
    return out


def load_zillow_neighbourhood(path, months):
    """Optional market data: Zillow neighbourhood-level ZHVI (home value index) CSV.
    Returns {neighbourhood: {month: value}} for Boston rows, or None if the file is absent."""
    p = Path(path)
    if not p.exists():
        return None
    z = pd.read_csv(p)
    z = z[(z["City"] == "Boston") & (z["State"] == "MA")]
    keep = [c for c in z.columns if c[:7] in months]
    return z.set_index("RegionName")[keep]
