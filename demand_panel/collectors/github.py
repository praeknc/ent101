"""GitHub: monthly counts of new public repositories mentioning a provider.

Uses the repository search API (`total_count` with a `created:` range), which
gives a monthly flow back to 2008. Code search would match imports more
precisely but has no date dimension, so it cannot build a panel.

Unauthenticated: 10 search requests/min. Set GITHUB_TOKEN for 30/min.
Queries live in data/reference/github_queries.csv; the `__all__` row is the
normaliser (all new repos that month).
"""
from __future__ import annotations

import time

import pandas as pd

from ..schema import make_obs
from ._http import get_json, github_headers

SEARCH_URL = "https://api.github.com/search/repositories"


def month_ranges(start: str, end: str):
    for p in pd.period_range(start, end, freq="M"):
        yield p.start_time.normalize(), p.end_time.normalize()


def count_repos(query: str, d0, d1) -> tuple[int, bool]:
    q = f"{query} created:{d0:%Y-%m-%d}..{d1:%Y-%m-%d}".strip()
    data = get_json(SEARCH_URL, params={"q": q, "per_page": 1}, headers=github_headers())
    return int(data["total_count"]), bool(data.get("incomplete_results"))


def collect(queries: pd.DataFrame, start: str, end: str, sleep: float | None = None) -> pd.DataFrame:
    authed = "Authorization" in github_headers()
    sleep = sleep if sleep is not None else (2.1 if authed else 6.5)
    rows = []
    for _, q in queries.fillna("").iterrows():
        is_all = q["provider_id"] == "__all__"
        for d0, d1 in month_ranges(start, end):
            n, incomplete = count_repos(q["query"] if not is_all else "is:public", d0, d1)
            rows.append(dict(
                source_id="github",
                entity_type="market" if is_all else "provider",
                entity_id=q["label"], provider_id="" if is_all else q["provider_id"],
                period_start=d0, period_end=d1, native_freq="M",
                variable="gh_new_repos_all" if is_all else "gh_new_repos", value=n,
                is_estimate=incomplete,
                source_url=SEARCH_URL, note=f"q={q['query'] or 'is:public'}"
                + ("; incomplete_results" if incomplete else "")))
            time.sleep(sleep)
    return make_obs(rows)
