# Demand panel coverage report

Window 2025-01 to 2026-11, frequency M. Generated from 24 observations.

## By source

| source_id | tier | first_period | last_period | n_periods | n_entities | n_obs | share_estimate | period_coverage |
|---|---|---|---|---|---|---|---|---|
| huggingface | A | 2026-11 | 2026-11 | 1 | 6 | 24 | 0.042 | 0.043 |
| openrouter | A |  |  | 0 | 0 | 0 |  | 0.0 |
| github | B |  |  | 0 | 0 | 0 |  | 0.0 |
| pypi | B |  |  | 0 | 0 | 0 |  | 0.0 |
| npm | B |  |  | 0 | 0 | 0 |  | 0.0 |
| provider_disclosures | B |  |  | 0 | 0 | 0 |  | 0.0 |
| china_demand | C |  |  | 0 | 0 | 0 |  | 0.0 |
| external_benchmarks | C |  |  | 0 | 0 | 0 |  | 0.0 |

## Periods observed per model and source

| model_id | provider_id | origin_country | open_weights | huggingface |
|---|---|---|---|---|
| openai/gpt-4o | openai | US | False | 0 |
| openai/gpt-4o-mini | openai | US | False | 0 |
| anthropic/claude-3.5-sonnet | anthropic | US | False | 0 |
| anthropic/claude-3.7-sonnet | anthropic | US | False | 0 |
| google/gemini-2.0-flash | google | US | False | 0 |
| meta/llama-3.1-8b-instruct | meta | US | True | 1 |
| meta/llama-3.3-70b-instruct | meta | US | True | 1 |
| mistral/mistral-7b-instruct | mistral | FR | True | 1 |
| deepseek/deepseek-v3 | deepseek | CN | True | 1 |
| deepseek/deepseek-r1 | deepseek | CN | True | 1 |
| qwen/qwen2.5-72b-instruct | alibaba | CN | True | 1 |

## Chinese demand gaps

| provider_id | periods | gaps | gap_share |
|---|---|---|---|
| alibaba | 23 | 23 | 1.0 |
| deepseek | 23 | 23 | 1.0 |

Cells with no Chinese-source observation but some other source (the only evidence on CN demand in that period is non-Chinese):

| provider_id | period | n_cn_obs | other_sources | gap |
|---|---|---|---|---|
| alibaba | 2026-11 | 0.0 | huggingface | True |
| deepseek | 2026-11 | 0.0 | huggingface | True |
