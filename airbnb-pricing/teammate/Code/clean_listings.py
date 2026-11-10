"""Re-creation of the teammate's cleaning step (their clean_listings.py was not shared).

Follows Boston_Airbnb_Variable_Screening.xlsx: Boston September 2026 detailed listings, rows with a
positive price and at least one review (3,002 listings), 48 features, full-sample median imputation.
Writes Data/Processed/listings_clean.csv in the layout train_models.py expects.

Usage (from airbnb-pricing/teammate): python Code/clean_listings.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))
import team_features as tf  # noqa: E402

SRC = ROOT.parent / "data" / "detailed" / "listings_detailed_2026Q3_sep.csv.gz"
OUT = ROOT / "Data" / "Processed" / "listings_clean.csv"

d = pd.read_csv(SRC, low_memory=False)
d["price"] = pd.to_numeric(d["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
d = d[(d["price"] > 0) & (d["number_of_reviews"] > 0)].reset_index(drop=True)
X, _ = tf.team_features(d)
labels = pd.DataFrame({
    "neighbourhood_cleansed": d["neighbourhood_cleansed"],
    "neighbourhood_area": d["neighbourhood_cleansed"].map(tf.AREA_OF),
    "property_type": d["property_type"],
    "property_group": d["property_type"].map(tf.GROUP_OF).fillna("Other / Specialty"),
    "room_type": d["room_type"],
})
trace = d[["id", "host_id", "last_scraped", "price_quote_checkin_date", "price_quote_checkout_date", "first_review", "last_review"]]
out = pd.concat([trace, labels, d[["price"]], pd.DataFrame({"log_price": np.log(d["price"])}), X], axis=1)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} listings, {X.shape[1]} features")
