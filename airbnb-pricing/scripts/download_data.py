"""Downloads the raw data used by Phase 3 Steps 2b-5 (not committed: about 120 MB).

  python scripts/download_data.py

- Inside Airbnb Boston: calendar.csv.gz, reviews.csv.gz and listings.csv.gz for the four snapshots
- Zillow: neighbourhood home value index (ZHVI), filtered to Boston, and ZIP-level rent index (ZORI) for Boston ZIPs
"""
import io
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MARKET = ROOT / "data" / "market"
SNAPSHOTS = {"2025Q4": "2025-12-27", "2026Q1": "2026-03-28", "2026Q2": "2026-06-15", "2026Q3": "2026-09-19"}
BASE = "https://data.insideairbnb.com/united-states/ma/boston/{date}/data/{file}"
ZHVI = "https://files.zillowstatic.com/research/public_csvs/zhvi/Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
ZORI = "https://files.zillowstatic.com/research/public_csvs/zori/Zip_zori_uc_sfrcondomfr_sm_month.csv"


def fetch(url, dest):
    if dest.exists():
        print("have", dest.relative_to(ROOT)); return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())
    print("got ", dest.relative_to(ROOT), f"{dest.stat().st_size / 1e6:.1f} MB")


for q, date in SNAPSHOTS.items():
    for f in ("calendar.csv.gz", "reviews.csv.gz", "listings.csv.gz"):
        fetch(BASE.format(date=date, file=f), RAW / q / f)

MARKET.mkdir(parents=True, exist_ok=True)
if not (MARKET / "Neighborhood_zhvi.csv").exists():
    with urllib.request.urlopen(ZHVI, timeout=300) as r:
        z = pd.read_csv(io.BytesIO(r.read()))
    z[(z["City"] == "Boston") & (z["State"] == "MA")].to_csv(MARKET / "Neighborhood_zhvi.csv", index=False)
    print("got  data/market/Neighborhood_zhvi.csv (Boston rows)")
if not (MARKET / "Zip_zori.csv").exists():
    with urllib.request.urlopen(ZORI, timeout=300) as r:
        z = pd.read_csv(io.BytesIO(r.read()))
    z[(z["State"] == "MA") & (z["Metro"].astype(str).str.contains("Boston"))].to_csv(MARKET / "Zip_zori.csv", index=False)
    print("got  data/market/Zip_zori.csv (Boston-metro ZIPs)")
