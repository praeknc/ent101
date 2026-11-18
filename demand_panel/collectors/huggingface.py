"""Hugging Face Hub: downloads, likes and derivative counts for open-weight models.

The Hub API has no download history. Each call returns the rolling 30-day
count and the all-time total *as of today*. Run `collect` on a schedule
(weekly), append to the store, and the panel derives period flows from
differences in `hf_downloads_alltime`.
"""
from __future__ import annotations

import pandas as pd

from ..crosswalk import Crosswalk
from ..schema import make_obs, now_utc
from ._http import get_json, hf_headers

MODEL_URL = "https://huggingface.co/api/models/{repo}"
LIST_URL = "https://huggingface.co/api/models"


def count_derivatives(repo: str, max_pages: int = 5, page_size: int = 1000) -> tuple[int, bool]:
    """Repos tagged base_model:<repo>. Returns (count, is_lower_bound)."""
    n, url, params = 0, LIST_URL, {"filter": f"base_model:{repo}", "limit": page_size}
    for _ in range(max_pages):
        data, resp = get_json(url, params=params, headers=hf_headers(), return_response=True)
        n += len(data)
        nxt = resp.links.get("next", {}).get("url")
        if not nxt:
            return n, False
        url, params = nxt, None
    return n, True


def collect(cw: Crosswalk, derivatives: bool = True, max_pages: int = 5) -> pd.DataFrame:
    today = pd.Timestamp(now_utc()[:10])
    rows = []
    for _, m in cw.hf_repos().iterrows():
        repo = m["hf_repo_id"]
        url = MODEL_URL.format(repo=repo)
        d = get_json(url, params=[("expand[]", "downloads"), ("expand[]", "downloadsAllTime"),
                                  ("expand[]", "likes")], headers=hf_headers())
        base = dict(source_id="huggingface", entity_type="model", entity_id=repo,
                    model_id=m["model_id"], provider_id=m["provider_id"],
                    period_end=today, native_freq="snapshot", source_url=url)
        rows.append({**base, "period_start": today - pd.Timedelta(days=29),
                     "variable": "hf_downloads_30d", "value": d.get("downloads", 0)})
        rows.append({**base, "period_start": today, "variable": "hf_downloads_alltime",
                     "value": d.get("downloadsAllTime", 0)})
        rows.append({**base, "period_start": today, "variable": "hf_likes", "value": d.get("likes", 0)})
        if derivatives:
            n, lb = count_derivatives(repo, max_pages=max_pages)
            rows.append({**base, "period_start": today, "variable": "hf_n_derivatives",
                         "value": n, "is_estimate": lb,
                         "note": "lower bound (page cap hit)" if lb else ""})
    return make_obs(rows)
