"""The teammate's cleaning pipeline (Boston_Airbnb_Variable_Screening.xlsx), re-implemented.

Their clean_listings.py was not shared, so this follows the spreadsheet's rules column by column:
48 features = 6 neighbourhood areas + 6 property groups + 4 room types + 5 size/bath
+ 10 amenity flags + 4 host + 7 review scores + 4 review history + 2 booking rules.
`train_models.py` (scripts/teammate/) consumes the same 48 columns.
"""
import numpy as np
import pandas as pd

NEIGHBOURHOOD_AREA = {
    "Downtown / Historic Core": ["Downtown", "Beacon Hill", "North End", "Chinatown", "West End", "Leather District"],
    "Premium Central / Waterfront": ["Back Bay", "South End", "Bay Village", "South Boston Waterfront"],
    "University / Medical / Cultural": ["Fenway", "Mission Hill", "Longwood Medical Area", "Allston"],
    "Inner Residential / Mixed-use": ["South Boston", "Charlestown", "East Boston", "Jamaica Plain"],
    "Outer Residential – West": ["Brighton", "Roslindale", "West Roxbury"],
    "Outer Residential – South": ["Dorchester", "Roxbury", "Mattapan", "Hyde Park"],
}
AREA_OF = {h: a for a, hs in NEIGHBOURHOOD_AREA.items() for h in hs}
AREA_COL = {
    "Downtown / Historic Core": "nbhd_downtown_historic_core",
    "Premium Central / Waterfront": "nbhd_premium_central_waterfront",
    "University / Medical / Cultural": "nbhd_university_medical_cultural",
    "Inner Residential / Mixed-use": "nbhd_inner_residential_mixed",
    "Outer Residential – West": "nbhd_outer_residential_west",
    "Outer Residential – South": "nbhd_outer_residential_south",
}

PROPERTY_GROUP = {
    "Apartment / Condo": ["Entire rental unit", "Private room in rental unit", "Shared room in rental unit", "Entire condo",
                          "Private room in condo", "Entire loft", "Private room in loft"],
    "House / Townhouse": ["Entire home", "Private room in home", "Shared room in home", "Entire townhouse",
                          "Private room in townhouse", "Private room in villa"],
    "Serviced accommodation": ["Entire serviced apartment", "Private room in serviced apartment", "Room in aparthotel"],
    "Hotel / B&B": ["Room in hotel", "Room in boutique hotel", "Private room in bed and breakfast"],
    "Guest accommodation": ["Entire guest suite", "Private room in guest suite", "Entire guesthouse"],
    "Other / Specialty": ["Houseboat", "Boat", "Private room in boat", "Entire place"],
}
GROUP_OF = {t: g for g, ts in PROPERTY_GROUP.items() for t in ts}
PROP_COL = {
    "Apartment / Condo": "prop_apartment_condo", "House / Townhouse": "prop_house_townhouse",
    "Serviced accommodation": "prop_serviced", "Hotel / B&B": "prop_hotel_bnb",
    "Guest accommodation": "prop_guest_accommodation", "Other / Specialty": "prop_other_specialty",
}
ROOM_COL = {"Entire home/apt": "room_entire_home_apt", "Private room": "room_private_room",
            "Hotel room": "room_hotel_room", "Shared room": "room_shared_room"}

# Substring roots from the "Amenities Mapping + Encoding" sheet (case-insensitive)
AMENITY_ROOTS = {
    "has_wifi": ["wifi", "wi-fi", "ethernet connection", "internet"],
    "has_kitchen": ["kitchen", "kitchenette"],
    "has_air_conditioning": ["air conditioning", "ac unit", "window ac", "central ac"],
    "has_parking": ["parking", "garage"],
    "has_laundry": ["washer", "dryer", "laundry"],          # dishwasher / hair dryer excluded below
    "has_workspace": ["dedicated workspace", "workspace"],
    "has_tv": ["tv", "television", "hdtv"],
    "has_coffee_facilities": ["coffee maker", "espresso machine", "keurig", "nespresso"],
    "has_outdoor_space": ["patio", "balcony", "backyard", "garden", "terrace", "outdoor dining", "outdoor furniture"],
    "allows_pets": ["pets allowed", "pet friendly"],
}
REVIEW_SCORES = ["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
                 "review_scores_communication", "review_scores_location", "review_scores_value"]
MEDIAN_IMPUTE = ["accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights", "number_of_reviews",
                 "number_of_reviews_ltm", "months_since_first_review", "days_since_last_review"] + REVIEW_SCORES


def _amenity_flags(raw):
    items = [a.strip().strip('"').lower() for a in str(raw).strip("[]").split('", "') if a.strip()]
    out = {}
    for col, roots in AMENITY_ROOTS.items():
        hit = False
        for a in items:
            if col == "has_laundry":
                a = a.replace("dishwasher", "").replace("hair dryer", "")
            if any(r in a for r in roots):
                hit = True
                break
        out[col] = int(hit)
    return out


def _baths(text):
    t = str(text).lower()
    if t in ("nan", ""):
        return np.nan, 0
    if "half" in t:
        return 0.5, int("shared" in t)
    num = pd.to_numeric(pd.Series([t.split(" ")[0]]), errors="coerce").iloc[0]
    return num, int("shared" in t)


def team_features(d, impute_medians=None):
    """Returns (X with the 48 teammate features, medians used). Pass impute_medians to reuse
    medians computed on a training set; by default they come from `d` itself, which is what
    the teammate's cleaning step does (full-sample medians)."""
    X = pd.DataFrame(index=d.index)
    area = d["neighbourhood_cleansed"].map(AREA_OF)
    for a, col in AREA_COL.items():
        X[col] = (area == a).astype(int)
    grp = d["property_type"].map(GROUP_OF).fillna("Other / Specialty")
    for g, col in PROP_COL.items():
        X[col] = (grp == g).astype(int)
    for r, col in ROOM_COL.items():
        X[col] = (d["room_type"] == r).astype(int)
    X["accommodates"] = d["accommodates"].astype(float)
    X["bedrooms"] = d["bedrooms"].astype(float)
    X["beds"] = d["beds"].astype(float)
    b = d["bathrooms_text"].map(_baths)
    X["bathrooms"] = [v for v, _ in b]
    X["bathrooms_share"] = [s for _, s in b]
    X = pd.concat([X, pd.DataFrame([_amenity_flags(a) for a in d["amenities"]], index=d.index)], axis=1)
    X["host_experience_months"] = 12 * d["hosts_time_as_host_years"].fillna(0) + d["hosts_time_as_host_months"].fillna(0)
    X["host_is_superhost"] = (d["host_is_superhost"] == "t").astype(int)
    X["host_identity_verified"] = (d["host_identity_verified"] == "t").astype(int)
    X["log_host_listings_count"] = np.log1p(d["host_listings_count"].fillna(1))
    for c in REVIEW_SCORES:
        X[c] = d[c].astype(float)
    X["number_of_reviews"] = d["number_of_reviews"].astype(float)
    X["number_of_reviews_ltm"] = d["number_of_reviews_ltm"].astype(float)
    scraped = pd.to_datetime(d["last_scraped"])
    X["months_since_first_review"] = (scraped - pd.to_datetime(d["first_review"])).dt.days / 30.44
    X["days_since_last_review"] = (scraped - pd.to_datetime(d["last_review"])).dt.days.astype(float)
    X["minimum_nights"] = d["minimum_nights"].astype(float)
    X["stay_length"] = (pd.to_datetime(d["price_quote_checkout_date"]) - pd.to_datetime(d["price_quote_checkin_date"])).dt.days.astype(float)
    med = X[MEDIAN_IMPUTE + ["stay_length"]].median() if impute_medians is None else impute_medians
    X = X.fillna(med)
    assert X.shape[1] == 48, X.shape
    return X, med


def extra_features(d):
    """The teammate's genuinely new ideas, as add-on columns for our pipeline (NaN kept for trees)."""
    X = pd.DataFrame(index=d.index)
    scraped = pd.to_datetime(d["last_scraped"])
    X["host_experience_months"] = 12 * d["hosts_time_as_host_years"] + d["hosts_time_as_host_months"]
    X["months_since_first_review"] = (scraped - pd.to_datetime(d["first_review"])).dt.days / 30.44
    ci, co = pd.to_datetime(d["price_quote_checkin_date"]), pd.to_datetime(d["price_quote_checkout_date"])
    X["stay_length"] = (co - ci).dt.days.astype(float)
    X["log_host_listings_count"] = np.log1p(d["host_listings_count"])
    return X
