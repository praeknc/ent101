"""Downloads a 320-px thumbnail of each listing's cover photo (June + September) for Step 5.

  python scripts/download_photos.py

Saves data/raw/photos/<sha1 of url>.jpg (not committed). Skips files already present.
"""
import hashlib
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "photos"
OUT.mkdir(parents=True, exist_ok=True)


def thumb(url):
    return url.replace("muscache.com/pictures/", "muscache.com/im/pictures/") + "?im_w=320"


def photo_path(url):
    return OUT / (hashlib.sha1(str(url).encode()).hexdigest() + ".jpg")


def get(url):
    dest = photo_path(url)
    if dest.exists():
        return "have"
    try:
        req = urllib.request.Request(thumb(url), headers={"User-Agent": "airbnb-pricing-research"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return "got"
    except Exception as e:  # noqa: BLE001 - a missing photo just becomes a missing feature
        return "fail"


if __name__ == "__main__":
    urls = set()
    for f in ["listings_detailed_2026Q2_jun.csv.gz", "listings_detailed_2026Q3_sep.csv.gz"]:
        urls |= set(pd.read_csv(ROOT / "data" / "detailed" / f, usecols=["picture_url"])["picture_url"].dropna())
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(get, sorted(urls)))
    print({k: res.count(k) for k in set(res)}, "of", len(urls), file=sys.stderr)
