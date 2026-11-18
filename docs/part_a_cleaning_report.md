# Part A price panel: cleaning report

Inputs: AI_API_price_panel.xlsx, prices_live.csv

Raw rows: 3,155. Chosen observations (model x market x date): 1,082. Model-months in panel: 3,356 across 325 models, 2023-09 to 2026-11.

Unlabelled-tier groups (archived OpenAI 2026 pages): 87 resolved against a labelled standard price, 31 unresolved.

## Row outcomes

| reason | rows |
|---|---|
| kept_candidate | 1271 |
| non_text_modality | 774 |
| non_standard_tier | 291 |
| higher_context_tier | 206 |
| removal_event | 167 |
| non_preferred_region | 165 |
| dated_variant_of_key | 152 |
| tier_unresolved | 61 |
| not_a_model_key | 27 |
| zero_price | 26 |
| other_host | 14 |
| priced_per_character | 1 |

| source | dated_variant_of_key | higher_context_tier | kept_candidate | non_preferred_region | non_standard_tier | non_text_modality | not_a_model_key | other_host | priced_per_character | removal_event | tier_unresolved | zero_price |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L | 61 | 0 | 328 | 4 | 3 | 237 | 0 | 6 | 0 | 167 | 0 | 26 |
| Live | 49 | 206 | 236 | 161 | 98 | 249 | 10 | 8 | 0 | 0 | 0 | 0 |
| W | 42 | 0 | 707 | 0 | 190 | 288 | 17 | 0 | 1 | 0 | 61 | 0 |

## Coverage by maker and market

| maker | market | models | model_months | first | last | share_third_party | share_carried |
|---|---|---|---|---|---|---|---|
| 01ai | CN | 1 | 7 | 2025-03 | 2025-09 | 1.0 | 0.86 |
| alibaba | CN | 13 | 175 | 2024-09 | 2026-07 | 0.0 | 0.49 |
| alibaba | INTL | 44 | 153 | 2025-09 | 2026-11 | 0.0 | 0.61 |
| anthropic | INTL | 36 | 450 | 2023-09 | 2026-11 | 0.0 | 0.71 |
| baidu | CN | 43 | 381 | 2024-02 | 2026-11 | 0.0 | 0.76 |
| deepseek | CN | 3 | 15 | 2025-03 | 2026-11 | 1.0 | 0.8 |
| deepseek | INTL | 6 | 15 | 2026-08 | 2026-11 | 1.0 | 0.27 |
| google | CN | 1 | 19 | 2024-03 | 2025-09 | 1.0 | 0.63 |
| google | INTL | 64 | 707 | 2024-06 | 2026-11 | 0.0 | 0.79 |
| meta | CN | 6 | 116 | 2024-02 | 2025-09 | 1.0 | 0.62 |
| minimax | INTL | 1 | 1 | 2026-11 | 2026-11 | 1.0 | 0.0 |
| mistral | CN | 1 | 20 | 2024-02 | 2025-09 | 1.0 | 0.6 |
| moonshot | INTL | 5 | 8 | 2026-08 | 2026-11 | 1.0 | 0.12 |
| openai | INTL | 91 | 1160 | 2023-09 | 2026-11 | 0.01 | 0.82 |
| other_open | CN | 5 | 99 | 2024-02 | 2025-09 | 1.0 | 0.62 |
| zhipu | CN | 2 | 21 | 2024-02 | 2026-11 | 1.0 | 0.57 |
| zhipu | INTL | 3 | 9 | 2026-08 | 2026-11 | 1.0 | 0.22 |

## Where each model-month price comes from

| price_source | model_months |
|---|---|
| L | 1621 |
| W | 1597 |
| Live | 138 |

| tier_resolution | model_months |
|---|---|
| labelled | 3113 |
| matched_to_labelled | 243 |

## Largest month-on-month cuts (check these by hand)

| model_id | period | price_change_pct | price_blended_3to1_usd_per_mtok | price_source | price_event | published_name |
|---|---|---|---|---|---|---|
| claude-instant-1-2 | 2023-11 | -0.9 | 0.26 | L | changed | claude-instant-1.2 |
| ernie-4-0-8k-ernie-4-0-8k-0613-ernie-4-0-8k-latest-ernie-4-0-8k-preview | 2025-03 | -0.844 | 0.9656104727352992 | W | snapshot | ERNIE-4.0-8K ERNIE-4.0-8K-0613 ERNIE-4.0-8K-Latest ERNIE-4.0-8K-Preview |
| ernie-3-5-128k | 2024-09 | -0.8 | 1.6958733747880175 | W | snapshot | ERNIE 3.5-128K |
| o3 | 2025-06 | -0.8 | 3.5 | L | changed | o3 |
| gemini-1-5-flash-latest | 2024-09 | -0.75 | 0.13124999999999998 | L | changed | gemini/gemini-1.5-flash-latest |
| gemini-1-5-flash | 2024-09 | -0.75 | 0.13124999999999998 | L | changed | gemini/gemini-1.5-flash |
| o1-mini | 2025-01 | -0.633 | 1.9250000000000003 | L | changed | o1-mini |
| gemini-1-5-pro | 2024-12 | -0.583 | 2.1875 | W | snapshot | Gemini 1.5 Pro |
| gpt-3-5-turbo | 2025-09 | -0.538 | 0.75 | L | changed | gpt-3.5-turbo |
| 通义千问-2-5-14b | 2025-09 | -0.5 | 0.2105706464518845 | W | snapshot | 通义千问 2.5-14B |
| 通义千问-2-5-7b | 2025-09 | -0.5 | 0.08773776935495187 | W | snapshot | 通义千问 2.5-7B |
| gemini-3-6-flash | 2026-08 | -0.5 | 1.5 | L | changed | gemini/gemini-3.6-flash |
| gpt-4o | 2024-11 | -0.417 | 4.375 | L | changed | gpt-4o |
| ernie-speed-appbuilder | 2024-04 | -0.375 | 0.6908558322049361 | W | snapshot | ERNIE Speed-AppBuilder |
| claude-sonnet-5 | 2026-07 | -0.333 | 4.0 | L | changed | claude-sonnet-5 |
| gpt-6-rosalind | 2026-11 | -0.333 | 20.0 | Live | live | gpt-6-rosalind (short context) |
| 通义千问-2-5-32b | 2025-09 | -0.314 | 0.4211412929037692 | W | snapshot | 通义千问 2.5-32B |
| gpt-5-6 | 2026-08 | -0.289 | 8.0 | L | changed | gpt-5.6 |
| gpt-5-6-sol | 2026-08 | -0.289 | 8.0 | L | changed | gpt-5.6-sol |
| claude-2 | 2023-11 | -0.27 | 12.0 | L | changed | claude-2 |
| ernie-speed-pro-128k | 2024-12 | -0.25 | 0.051506036507478654 | W | snapshot | ERNIE-Speed-Pro-128K |
| claude-haiku-3-5 | 2025-02 | -0.2 | 1.6 | L | changed | claude-3-5-haiku-20241022 |

## Largest month-on-month rises (often a source correcting an earlier error: check)

| model_id | period | price_change_pct | price_blended_3to1_usd_per_mtok | price_source | price_event | published_name |
|---|---|---|---|---|---|---|
| gemini-1-5-pro | 2024-07 | 9.0 | 5.25 | L | changed | gemini/gemini-1.5-pro |
| gemini-1-5-pro-latest | 2024-07 | 4.5 | 2.8875 | L | changed | gemini/gemini-1.5-pro-latest |
| gemini-flash-lite-latest | 2026-09 | 3.857 | 0.85 | L | changed | gemini/gemini-flash-lite-latest |
| ernie-lite-pro-128k | 2024-12 | 2.333 | 0.034337357671652476 | W | snapshot | ERNIE-Lite-Pro-128K |
| gemini-2-5-flash-preview | 2025-11 | 2.238 | 0.85 | W | snapshot | Gemini 2.5 Flash Preview |
| gemini-2-5-flash-preview-05-20 | 2025-06 | 2.238 | 0.85 | L | changed | gemini/gemini-2.5-flash-preview-05-20 |
| deepseek-v4-flash-0731 | 2026-11 | 1.592 | 0.648 | Live | live | deepseek-v4-flash-0731 |
| gemini-flash-latest | 2026-09 | 0.765 | 1.5 | L | changed | gemini/gemini-flash-latest |
| gemini-pro-latest | 2026-09 | 0.309 | 4.5 | L | changed | gemini-pro-latest |

## Rules

- Standard on-demand tier, base context-length tier, non-thinking mode where both exist.
- Text models only (vision/OCR/omni, audio, image, embedding, moderation and tool endpoints excluded).
- Market: model uses INTL prices if it has any, else CN (RMB converted at the workbook's monthly FX).
- Region order within INTL: International/Singapore, then US, then Global deployments, then Hong Kong.
- Source order on the same date: archived page (W) and live page over LiteLLM (L).
- LiteLLM only moves prices on first_seen/changed events; cache_changed updates cached input only.
- Archived/live snapshots carried forward at most 6 months; a LiteLLM price stays in force until its next change event or removal (LiteLLM is a change log). Never past a LiteLLM removal.
- Blended price = (3 x input + 1 x output) / 4 per 1M tokens.
- Price changes use the local list currency (RMB prices do not move with the exchange rate).
- Index: chained matched-model Jevons on the local blended price; '-latest' alias pointers and rises above +300% or exact x10 jumps either way (`suspect_change`: a source correcting itself or a per-1K/per-1M slip) are left out of the index but kept in the panel.
