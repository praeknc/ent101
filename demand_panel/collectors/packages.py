"""PyPI and npm download counts for provider SDK packages.

PyPI: pypistats.org JSON API (about 180 days of daily data, mirrors excluded).
      For longer history run `PYPI_BIGQUERY_SQL` on BigQuery's public
      `bigquery-public-data.pypi.file_downloads` table and load the result with
      `load_pypi_bigquery`.
npm:  api.npmjs.org range endpoint (daily; max 18 months per request, so
      requests are chunked).
"""
from __future__ import annotations

import urllib.parse

import pandas as pd

from ..crosswalk import Crosswalk
from ..schema import make_obs
from ._http import get_json

PYPISTATS_URL = "https://pypistats.org/api/packages/{pkg}/overall"
NPM_URL = "https://api.npmjs.org/downloads/range/{start}:{end}/{pkg}"

PYPI_BIGQUERY_SQL = """
-- Daily downloads per SDK package, excluding obvious CI/mirror installers.
SELECT DATE(timestamp) AS date, file.project AS package, COUNT(*) AS downloads
FROM `bigquery-public-data.pypi.file_downloads`
WHERE file.project IN UNNEST(@packages)
  AND DATE(timestamp) BETWEEN @start AND @end
  AND details.installer.name IN ('pip', 'uv', 'poetry', 'pdm')
GROUP BY date, package
ORDER BY date, package
"""


def _rows(eco: str, var: str, pkg: str, provider: str, daily: pd.DataFrame, url: str) -> list[dict]:
    return [dict(source_id=eco, entity_type="package", entity_id=pkg, provider_id=provider,
                 period_start=d, period_end=d, native_freq="D", variable=var, value=v,
                 source_url=url) for d, v in zip(daily["date"], daily["downloads"])]


def collect_pypi(cw: Crosswalk) -> pd.DataFrame:
    rows = []
    for _, p in cw.packages_for("pypi").iterrows():
        url = PYPISTATS_URL.format(pkg=p["package"])
        data = get_json(url, params={"mirrors": "false"})["data"]
        df = pd.DataFrame(data)
        if "category" in df:
            df = df[df["category"] == "without_mirrors"]
        rows += _rows("pypi", "pypi_downloads", p["package"], p["provider_id"], df, url)
    return make_obs(rows)


def load_pypi_bigquery(path, cw: Crosswalk) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    for pkg, g in df.groupby("package"):
        rows += _rows("pypi", "pypi_downloads", pkg, cw.provider_of_package("pypi", pkg), g,
                      "bigquery-public-data.pypi.file_downloads")
    return make_obs(rows)


def collect_npm(cw: Crosswalk, start: str, end: str) -> pd.DataFrame:
    rows = []
    # 18-month cap per request -> chunk by 365 days
    edges = list(pd.date_range(start, end, freq="365D")) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    for _, p in cw.packages_for("npm").iterrows():
        enc = urllib.parse.quote(p["package"], safe="@")
        for a, b in zip(edges[:-1], edges[1:]):
            b = b - pd.Timedelta(days=1)
            url = NPM_URL.format(start=f"{a:%Y-%m-%d}", end=f"{b:%Y-%m-%d}", pkg=enc)
            data = get_json(url)
            df = pd.DataFrame(data.get("downloads", [])).rename(columns={"day": "date"})
            if not df.empty:
                rows += _rows("npm", "npm_downloads", p["package"], p["provider_id"], df, url)
    return make_obs(rows)
