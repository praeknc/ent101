#!/usr/bin/env python3
"""
litellm_history.py  -  backup historical price panel from LiteLLM's price file

LiteLLM (github.com/BerriAI/litellm) keeps one JSON file of per-token prices,
model_prices_and_context_window.json, and its git history records every price
change with a date. This script walks that history and writes one row each time
a model first appears, changes price, or disappears.

SECONDARY SOURCE: community-maintained, not the providers' own pages. Use it to
fill gaps and cross-check the Wayback / official snapshots, not as the sole
citation. Every row is marked needs_review=True for that reason.

DATES ARE WHEN LITELLM UPDATED THE FILE, not when the provider changed the
price. Usually same day or within days for big launches, but some updates lag
by months (e.g. a price cut recorded long after it happened). Confirm a
change's date against an official announcement or Wayback snapshot.
Rows marked SUSPECT were reverted/corrected soon after (LiteLLM data errors).

COVERAGE
  * from Sep 2023 (when the file got its current name) to today
  * openai (+ text-completion-openai), anthropic, google (Gemini API), alibaba
    (dashscope / qwencloud / qwen_ai_platform); Baidu has no first-party entry
  * base per-token prices only: long-context tiers (*_above_200k_tokens etc.),
    batch and per-image/per-second prices are not columns here (raw_row keeps
    the batch price when present)

SETUP
  needs only git and Python 3 (no pip packages)

USAGE
  python3 litellm_history.py                       # all providers, daily sampling
  python3 litellm_history.py --provider openai anthropic
  python3 litellm_history.py --sample all          # every commit, not one per day

OUTPUT (./pricing_data/)
  prices_litellm.csv     event rows (first_seen / changed / removed)
  litellm_repo/          partial git clone (history only, ~100 MB)
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_URL = "https://github.com/BerriAI/litellm"
PRICE_FILE = "model_prices_and_context_window.json"
OUT = Path("pricing_data")
CSV_PATH = OUT / "prices_litellm.csv"

# our provider name -> LiteLLM 'litellm_provider' values (first-party APIs only)
PROVIDERS = {
    "openai": {"openai", "text-completion-openai"},
    "anthropic": {"anthropic"},
    "google": {"gemini"},
    "alibaba": {"dashscope", "qwencloud", "qwen_ai_platform"},
    "baidu": set(),  # only resellers (fireworks, novita, ...) list ERNIE models
}

FIELDS = ["provider", "model", "section", "input_per_1m", "output_per_1m", "currency",
          "unit_detected", "needs_review", "review_reason", "raw_row", "source_url",
          "source_type", "retrieved_at", "snapshot_sha256",
          # extra columns
          "event", "cache_read_per_1m", "litellm_provider"]


def git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       check=True, text=not binary)
    return r.stdout


def ensure_repo(repo):
    """Partial clone: full commit history, file contents fetched on demand."""
    if (repo / ".git").exists():
        print(f"Updating {repo} ...")
        git(repo, "fetch", "-q", "--filter=blob:none", "origin")
        return
    print(f"Cloning LiteLLM history into {repo} (about 100 MB, a few minutes) ...")
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                    "--single-branch", REPO_URL, str(repo)], check=True)


def versions(repo, since, sample):
    """[(commit_sha, iso_date, blob_oid)] oldest first; one per day unless sample=all."""
    log = git(repo, "log", "--format=%H %cI", "origin/HEAD", "--", PRICE_FILE).split("\n")
    rows, seen_days = [], set()
    for line in log:  # newest first
        if not line.strip():
            continue
        sha, date = line.split()
        if date[:10] < since:
            break
        if sample == "daily":
            if date[:10] in seen_days:
                continue  # keep the last commit of each day
            seen_days.add(date[:10])
        rows.append((sha, date))
    rows.reverse()
    out = []
    for sha, date in rows:
        try:
            oid = git(repo, "rev-parse", f"{sha}:{PRICE_FILE}").strip()
        except subprocess.CalledProcessError:
            continue
        out.append((sha, date, oid))
    return out


def prefetch(repo, oids, chunk=200):
    """Download file versions in batches instead of one request per version."""
    for i in range(0, len(oids), chunk):
        part = oids[i:i + chunk]
        print(f"  downloading versions {i + 1}-{i + len(part)} of {len(oids)}")
        subprocess.run(["git", "-C", str(repo), "-c", "fetch.negotiationAlgorithm=noop",
                        "fetch", "-q", "--no-tags", "--no-write-fetch-head",
                        "--filter=blob:none", "origin", *part], check=True)


def per_1m(x):
    return None if x is None else round(float(x) * 1e6, 6)


def snapshot(data, wanted):
    """{(provider, model): (price tuple, info)} for one version of the file."""
    lp_to_ours = {lp: ours for ours, lps in PROVIDERS.items() if ours in wanted for lp in lps}
    out = {}
    for model, v in data.items():
        if not isinstance(v, dict):
            continue
        lp = v.get("litellm_provider")
        if lp not in lp_to_ours:
            continue
        pin, pout = v.get("input_cost_per_token"), v.get("output_cost_per_token")
        if pin is None and pout is None:
            continue  # per-image / per-second / per-character models
        try:
            prices = (per_1m(pin), per_1m(pout), per_1m(v.get("cache_read_input_token_cost")))
        except (TypeError, ValueError):
            continue
        extra = {k: v[k] for k in ("input_cost_per_token_batches", "output_cost_per_token_batches",
                                   "input_cost_per_token_above_200k_tokens",
                                   "input_cost_per_token_above_128k_tokens", "source")
                 if k in v}
        out[(lp_to_ours[lp], model)] = (prices, lp, v.get("mode", ""), extra)
    return out


def row(provider, model, prices, lp, mode, extra, event, sha, date):
    pin, pout, pcache = prices
    reasons = ["secondary source (LiteLLM community price file)"]
    if pin is None or pout is None:
        reasons.append("input or output price missing")
    if extra and any(k.endswith(("128k_tokens", "200k_tokens")) for k in extra):
        reasons.append("has long-context tier; base tier shown")
    return dict(
        provider=provider, model=model, section=mode,
        input_per_1m="" if pin is None else pin, output_per_1m="" if pout is None else pout,
        currency="USD", unit_detected="1M", needs_review=True,
        review_reason="; ".join(reasons), raw_row=json.dumps(extra) if extra else "",
        source_url=f"{REPO_URL}/blob/{sha}/{PRICE_FILE}", source_type="litellm_git",
        retrieved_at=date, snapshot_sha256=sha, event=event,
        cache_read_per_1m="" if pcache is None else pcache, litellm_provider=lp)


def _days(a, b):
    from datetime import date
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days


def flag_suspect(rows, window=21):
    """Mark LiteLLM data errors: a price that is reverted within `window` days,
    a model removed and re-added within `window` days, and zero prices."""
    by_model = {}
    for r in rows:
        by_model.setdefault((r["provider"], r["model"]), []).append(r)

    def add(r, why):
        r["review_reason"] += "; " + why

    for evs in by_model.values():
        for i in range(1, len(evs) - 1):
            prev, cur, nxt = evs[i - 1], evs[i], evs[i + 1]
            same = lambda a, b: (a["input_per_1m"], a["output_per_1m"]) == (b["input_per_1m"], b["output_per_1m"])
            if (cur["event"] == "changed" and nxt["event"] == "changed" and same(prev, nxt)
                    and _days(cur["retrieved_at"], nxt["retrieved_at"]) <= window):
                add(cur, f"SUSPECT: reverted on {nxt['retrieved_at'][:10]} (likely LiteLLM error)")
            if (cur["event"] == "removed" and nxt["event"] == "first_seen"
                    and _days(cur["retrieved_at"], nxt["retrieved_at"]) <= window):
                add(cur, f"SUSPECT: re-added on {nxt['retrieved_at'][:10]}")
                add(nxt, "re-added after brief removal (not a new model)")
        if len(evs) >= 2 and evs[0]["event"] == "first_seen" and evs[1]["event"] == "changed" \
                and _days(evs[0]["retrieved_at"], evs[1]["retrieved_at"]) <= 3:
            add(evs[0], f"SUSPECT: corrected on {evs[1]['retrieved_at'][:10]}")
        for r in evs:
            if r["input_per_1m"] == 0.0 or r["output_per_1m"] == 0.0:
                add(r, "zero price (free preview or missing data)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", nargs="*", choices=list(PROVIDERS),
                    help="default: all")
    ap.add_argument("--since", default="2023-01", help="YYYY or YYYY-MM (default 2023-01)")
    ap.add_argument("--sample", choices=["daily", "all"], default="daily",
                    help="daily = last commit of each day (default); all = every commit")
    ap.add_argument("--repo-dir", default=str(OUT / "litellm_repo"))
    args = ap.parse_args()
    wanted = set(args.provider or PROVIDERS)
    since = (args.since + "-01-01")[:10] if len(args.since) == 4 else (args.since + "-01")[:10]
    repo = Path(args.repo_dir)

    ensure_repo(repo)
    vers = versions(repo, since, args.sample)
    if not vers:
        sys.exit("No versions of the price file found in that window.")
    print(f"{len(vers)} versions of {PRICE_FILE} from {vers[0][1][:10]} to {vers[-1][1][:10]}")
    prefetch(repo, sorted({oid for _, _, oid in vers}))

    rows, state, bad = [], {}, 0
    for sha, date, oid in vers:
        try:
            data = json.loads(git(repo, "cat-file", "blob", oid, binary=True))
        except (json.JSONDecodeError, UnicodeDecodeError, subprocess.CalledProcessError):
            bad += 1  # a commit that broke the JSON; the next one usually fixes it
            continue
        snap = snapshot(data, wanted)
        for key, (prices, lp, mode, extra) in snap.items():
            old = state.get(key)
            if old is None or old[0] != prices:
                event = ("first_seen" if old is None
                         else "changed" if old[0][:2] != prices[:2] else "cache_changed")
                rows.append(row(*key, prices, lp, mode, extra, event, sha, date))
        for key in state.keys() - snap.keys():
            prices, lp, mode, extra = state[key]
            rows.append(row(*key, prices, lp, mode, extra, "removed", sha, date))
        state = snap

    flag_suspect(rows)
    OUT.mkdir(exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} rows -> {CSV_PATH}" + (f"  ({bad} unreadable versions skipped)" if bad else ""))
    # coverage: models with a price change or first appearance, per provider x year
    years = sorted({r["retrieved_at"][:4] for r in rows}) or [since[:4]]
    first = Counter((r["provider"], r["retrieved_at"][:4]) for r in rows if r["event"] == "first_seen")
    changed = Counter((r["provider"], r["retrieved_at"][:4]) for r in rows if r["event"] == "changed")
    print("\nCOVERAGE  new models / price changes per year")
    print("  " + "provider".ljust(10) + "".join(y.rjust(10) for y in years))
    for p in sorted(wanted):
        note = "  (no first-party entries in LiteLLM)" if not PROVIDERS[p] else ""
        print("  " + p.ljust(10) + "".join(f"{first[(p, y)]}/{changed[(p, y)]}".rjust(10)
                                           for y in years) + note)
    print(f"\nLiteLLM history starts Sep 2023; earlier months must come from Wayback.")


if __name__ == "__main__":
    main()
