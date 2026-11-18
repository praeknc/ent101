# Demand-side data and panel construction (thesis §4.4)

This package builds the thesis's **second dataset**: demand for AI models, observed at the **model**, **application** and **developer/provider** level over time. It links to the Part A price and index panel by model and date.

```
collectors/*  ->  observations.csv (long, one schema)  ->  panel.py  ->  model / provider / app panels
                                                                          |
                                     Part A prices + index  ->  merge.py  ->  model_panel_merged_M.csv
```

## Units of analysis

| Panel | Key | Main sources | File |
|---|---|---|---|
| Model × period | `model_id`, `period` | OpenRouter tokens and shares, Hugging Face downloads, CN ranks | `data/processed/model_panel_{W,M}.csv` |
| Provider × period | `provider_id`, `period` | SDK downloads (PyPI, npm), GitHub repos, disclosures, CN announcements, model flows summed to provider | `provider_panel_{W,M}.csv` |
| App × model × period | `app_id`, `model_id`, `period` | OpenRouter app routing (each app's model mix) | `app_panel_{W,M}.csv` |

Each panel also has a `*_long.csv` version that keeps quality columns for every cell: `n_obs`, `sources`, best `tier`, and `any_estimate`. A missing value means *not observed*. It never means zero.

## Sources and tiers

The full table, with what each source can and cannot show and its sampling-bias notes, is in [`docs/source_tiers.md`](../docs/source_tiers.md). It is generated from `sources.py`.

| Source | Tier | Level | How collected |
|---|---|---|---|
| OpenRouter rankings | A | model, app | `collect openrouter` scrapes the rankings site's endpoints. **Run weekly** (see below) |
| Hugging Face Hub | A | model | API, snapshot only. **Run weekly**: history cannot be recovered later |
| GitHub new repos | B | provider | Repository search API, monthly `created:` ranges, normalised by all new repos |
| PyPI / npm SDK downloads | B | package → provider | pypistats (~180 days) or BigQuery SQL for long history; npm range API |
| Provider disclosures | B | provider | Manual template; quote and URL required |
| Chinese demand | C | provider / model | Manual template; gaps reported in `docs/coverage_report.md` |
| Adoption surveys, job postings | C | market / provider | Manual template; **validation only**, excluded from panels unless `--include-validation` |

Tier rules:
- **A** is a systematic log of behaviour and is used for main outcomes.
- **B** is an indirect proxy or an unaudited aggregate. It is used for robustness checks and as level anchors.
- **C** covers announcements, rankings and surveys. It is used for description and validation.

## OpenRouter: `python -m demand_panel collect openrouter`

This command takes about 4 minutes and saves raw JSON to `data/raw/openrouter/<YYYYMMDD>/`. It collects:

| Data | Coverage | Variables |
|---|---|---|
| Daily tokens, prompt/completion and requests for **every** model variant (standard / free / batch) | last ~31 days | `or_tokens`, `or_tokens_prompt`, `or_tokens_completion`, `or_requests`, `or_tokens_free` |
| Weekly rankings chart | ~53 weeks, top ~9 models per week plus "Others" | `or_tokens_topweekly`, `or_tokens_platform` (all models), derived `or_token_share_platform` |
| Top 20 apps | day / week / month snapshot | `or_app_tokens_day/week/month` |

It also writes a readable table to `data/processed/openrouter_weekly_table.csv`, one row per week × model.

OpenRouter models that aren't in the crosswalk are added automatically. Names are matched to Part A keys too, so `anthropic/claude-sonnet-4.5` links to the Part A key `claude-sonnet-4-5`.

Re-running the command replaces values for the same model × day instead of adding them twice, so overlapping weekly runs are safe. To re-parse a saved snapshot without downloading again, use `--from-raw data/raw/openrouter/<date>`.

## Merge keys

- **Model.** `data/reference/model_crosswalk.csv` maps one canonical `model_id` to the OpenRouter slug, the HF repo, the `part_a_model_id` and free-text aliases. OpenRouter variant suffixes (`:free`, `:beta`, …) are stripped for matching but kept in `note`. Any id that does not match is written to `data/processed/unmatched_ids.txt` and listed in the coverage report. Unmatched ids are never dropped silently.
- **Provider.** `provider_id` in the crosswalk. `package_map.csv` maps SDK packages to providers.
- **Date.** With the cleaned Part A panel (the default), the merge joins on the same month (`--how exact`), so the cleaner's carry-forward rules apply. For any other prices file, the default is **as of the period end**: the last price on or before the period end, within 92 days. Alternatively, `--how period_mean` averages prices inside the period. The merge adds `_lag1` columns (previous month, consecutive months only) and the overall and maker price indices.

### Part A prices: `python -m demand_panel part-a`

This command cleans the Part A workbook (`data/raw/part_a/AI_API_price_panel.xlsx`, sheet *Panel (USD)*) plus the raw live scrape (`data/raw/part_a/prices_live.csv`). The output is a **model × month price panel** that the merge uses by default. The code is in `part_a.py`, and every rule and its row counts are in [`docs/part_a_cleaning_report.md`](../docs/part_a_cleaning_report.md).

| Problem in the raw panel | Rule |
|---|---|
| Standard / Batch / Flex / Priority / Fast / fine-tune rows mixed together | Standard tier only. LiteLLM `fast/` entries count as priority |
| 2026 archived OpenAI pages have no tier labels | Pick the candidate closest to a labelled standard price for the same model within 400 days (±25%); otherwise drop |
| Dated snapshots (`gpt-4o-2024-05-13`) share a key with the alias | Use alias rows when a source has them |
| Alibaba regions with different prices; LiteLLM `us/` +10% entries | International/Singapore, then US, then Global, then Hong Kong; one region per model and source |
| Thinking vs non-thinking, long-context tiers | Non-thinking, base context tier |
| Resellers (Alibaba or Baidu selling DeepSeek/Kimi/GLM; `azure/`, `openrouter/`) | One host per model: the maker if it sells the model, else its main host (`third_party_host` flag) |
| RMB prices converted at monthly FX | `price_*_local` columns; price changes measured in the list currency |
| Image, audio, embedding, vision, moderation, tool endpoints | Dropped (text LLMs only) |
| LiteLLM lags (README says to prefer W over L) | W/Live beat L on the same date; L only moves prices on `first_seen`/`changed` events |
| Snapshot gaps | W/Live carried at most 6 months; L (a change log) carried until its next event or removal |
| `-latest` aliases and source corrections | Kept in the panel but flagged (`is_alias_pointer`, `suspect_change` for >+300% or exact ×10 jumps) and left out of the index |

Outputs go to `data/processed/part_a/`:

| File | Contents |
|---|---|
| `part_a_prices_M.csv` | One row per model × month. Columns: `price_in/out/cached_in_usd_per_mtok`, `price_blended_3to1_usd_per_mtok`, local-currency prices, `price_change_pct`, `price_cut`, `price_rise`, `months_since_obs`, `price_source`, `market`, `maker`, `host`, quality flags |
| `part_a_index_M.csv` | Overall index: chained matched-model Jevons, local currency |
| `part_a_index_maker_M.csv` | Index by maker |
| `part_a_index_market_M.csv` | Index by market |
| `part_a_observations.csv` | One observation per model × market × date, after the selection rules |
| `part_a_rows_classified.csv` | Every raw row with its `drop_reason` |

The command also adds every priced Part A model to `data/reference/model_crosswalk.csv`, with `model_id = maker/key` and `part_a_model_id = key`. Seed models get linked to their Part A key. To skip this, use `--no-sync-crosswalk`.

**Still needs a hand check:** the largest cuts and rises listed in the cleaning report. Claude Instant 1.2's −90% in 2023-11 and Gemini 1.5 Pro's +900% in 2024-07 are both flagged as LiteLLM unit slips.

### Part A input contract (for other price files)

```
prices.csv : model_id, date, <numeric price cols>    (model_id = Part A id)
index.csv  : date, [key cols e.g. provider_id/segment], <numeric index cols>
```

`merge_prices` returns diagnostics: the row match rate, models that are in demand but not in Part A, and models in Part A that are not in demand. `rollup_prices_to_provider` builds a provider × period price, simple or token-weighted, for the provider panel.

## Usage

```bash
pip install -r requirements.txt
python -m demand_panel docs                                   # source tiers + variable dictionary
python -m demand_panel templates                              # manual-entry CSV templates
python -m demand_panel collect huggingface                    # weekly snapshot
GITHUB_TOKEN=... python -m demand_panel collect github --start 2023-01 --end 2026-10
python -m demand_panel collect pypi
python -m demand_panel collect npm --start 2024-06-01 --end 2026-10-31
python -m demand_panel load openrouter data/raw/openrouter_export.csv --freq W
python -m demand_panel load manual                            # data/manual/*.csv
python -m demand_panel build --freq M --start 2023-01 --end 2026-11
python -m demand_panel part-a                                 # clean Part A -> model x month prices + index
python -m demand_panel merge                                  # demand panel + Part A prices + indices
python -m pytest tests
```

## Design rules

1. **One long schema** (`schema.py`). Every collector writes rows with `source_id`, `tier`, entity ids, `period_start`/`period_end`, `variable`, `value`, `is_estimate`, `retrieved_at`, `source_url` and `note`. The variable and source must be registered, and derived variables cannot be entered by hand.
2. **Frequency conversion follows the dictionary.** Each variable's `agg` rule (`sum`/`last`/`mean`/`max`) sets how it is converted. A flow that straddles two periods, such as a week crossing a month end, is split pro rata by days, so totals are conserved.
3. **Sparse disclosures are not interpolated.** They stay as point observations, used as level anchors.
4. **HF flows come from the all-time counter.** They are differences in `hf_downloads_alltime` between *consecutive* periods. If a snapshot is missing, the flow is left missing.
5. **Manual rows need provenance.** Each one needs a `source_url` plus a verbatim `quote` or a page/timestamp reference. Values are stored as stated.

## Known gaps to state in the thesis

- OpenRouter is a self-selected gateway sample. It does not show first-party API or consumer-app traffic, or usage inside China.
- Hugging Face has no download history before the first snapshot (2026-11-18 in this repo). Users who download from ModelScope or mirrors are not counted.
- The GitHub queries are keyword matches. Their precision has to be audited, and the audit columns are in `github_queries.csv`.
- Chinese demand depends on announcements. `docs/coverage_report.md` lists every CN provider × month with no Chinese-source observation.
