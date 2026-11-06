"""Writes scripts/feature_cases.json: raw listing inputs + the feature vectors Python builds for them.
scripts/verify_js.js checks that interface/pricing.js builds identical vectors from the same inputs."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pricing_features as pf

official = pd.read_csv(ROOT / "data/neighbourhoods.csv")["neighbourhood"]
hoods = pf.model_neighbourhoods(official)
rng = np.random.default_rng(0)
lic = {"licensed": "STR-123456", "exempt": "Exempt", "none": None}
cases = []
for i in range(60):
    room = ["Entire home/apt", "Private room", "Hotel room", "Shared room"][i % 4]
    hood = list(official)[i % len(official)]
    if hood == "Harbor Islands":
        hood = "Leather District"
    beds = int(rng.integers(0, 6)); reviews = int(rng.integers(0, 300)); status = ["licensed", "exempt", "none"][i % 3]
    raw = {"neighbourhood": hood, "room_type": room, "bedrooms": beds, "minimum_nights": int(rng.choice([1, 2, 3, 7, 30, 90, 400])),
           "number_of_reviews": reviews, "reviews_per_month": float(round(rng.uniform(0, 25), 2)),
           "number_of_reviews_ltm": int(rng.integers(0, 400)), "host_listings": int(rng.choice([1, 2, 3, 4, 10, 11, 50, 51, 300])),
           "availability_365": int(rng.integers(0, 366)), "license": status,
           "lat": float(42.33 + rng.uniform(-0.05, 0.05)), "lon": float(-71.07 + rng.uniform(-0.05, 0.05)),
           "kw": {k: bool(rng.integers(0, 2)) for k in pf.KEYWORDS}}
    # same path the notebook takes: a listings row -> precheck rules -> build_features
    title = ("studio" if beds == 0 else f"{beds}BR") + " " + " ".join(
        {"kw_parking": "parking", "kw_luxury": "modern", "kw_view": "view", "kw_amenities": "gym"}[k] for k, v in raw["kw"].items() if v)
    row = pd.DataFrame([{"id": i, "name": title, "host_id": i, "neighbourhood": hood, "room_type": room,
                         "latitude": raw["lat"], "longitude": raw["lon"], "price": 100.0,
                         "minimum_nights": raw["minimum_nights"], "number_of_reviews": reviews,
                         "reviews_per_month": raw["reviews_per_month"],
                         "number_of_reviews_ltm": min(raw["number_of_reviews_ltm"], reviews),
                         "calculated_host_listings_count": raw["host_listings"],
                         "availability_365": raw["availability_365"], "license": lic[status]}])
    geo = pf.load_geo(ROOT / "data/neighbourhoods.geojson")
    clean, _ = pf.precheck(row, official, geo)
    X = pf.build_features(clean, hoods)
    cases.append({"input": raw, "features": list(X.columns), "x": X.iloc[0].tolist()})
(ROOT / "scripts/feature_cases.json").write_text(json.dumps(cases))
print("wrote", len(cases), "cases")
