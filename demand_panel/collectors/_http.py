"""Small HTTP helper with retry/backoff. Uses only `requests`."""
from __future__ import annotations

import os
import time

import requests

USER_AGENT = "demand-panel-research/0.1"


def get_json(url: str, params=None, headers=None, retries: int = 4, backoff: float = 2.0,
             timeout: int = 30, return_response: bool = False):
    h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    h.update(headers or {})
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            limited = r.status_code == 429 or (r.status_code == 403 and "rate limit" in r.text.lower())
            if limited:
                wait = backoff * (2 ** attempt)
                reset, retry_after = r.headers.get("X-RateLimit-Reset"), r.headers.get("Retry-After")
                if reset and reset.isdigit():
                    wait = min(max(int(reset) - time.time(), 1), 120)
                elif retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return (r.json(), r) if return_response else r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def github_headers() -> dict:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def hf_headers() -> dict:
    tok = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}
