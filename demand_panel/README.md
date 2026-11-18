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
| OpenRouter rankings | A | model, model × app | CSV export/transcription (no public history API); `fetch_models()` for slugs and list prices |
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

## Merge keys

- **Model.** `data/reference/model_crosswalk.csv` maps one canonical `model_id` to the OpenRouter slug, the HF repo, the `part_a_model_id` and free-text aliases. OpenRouter variant suffixes (`:free`, `:beta`, …) are stripped for matching but kept in `note`. Any id that does not match is written to `data/processed/unmatched_ids.txt` and listed in the coverage report. Unmatched ids are never dropped silently.
- **Provider.** `provider_id` in the crosswalk. `package_map.csv` maps SDK packages to providers.
- **Date.** The demand period is matched to Part A **as of the period end**: the last price on or before the period end, within 92 days. The alternative, `--how period_mean`, averages Part A prices inside the period. The merge also adds a `_lag1` column for each price.

### Part A input contract

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
python -m demand_panel merge --prices ../partA/prices.csv --index ../partA/index.csv --index-by provider_id
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
