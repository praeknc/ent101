# Variable dictionary (demand-side panel)

Generated from `demand_panel/dictionary.py`. `agg` is the rule used to collapse native frequency to the panel frequency. Derived variables are computed in `panel.py`.

## openrouter (tier A)

| name | label | entity_level | unit | native_freq | agg | derived | definition |
|---|---|---|---|---|---|---|---|
| or_tokens | OpenRouter tokens (total) | model | tokens | W | sum | False | Prompt + completion tokens routed to the model in the period, summed across apps. |
| or_tokens_prompt | OpenRouter prompt tokens | model | tokens | W | sum | False | Input tokens, where the export separates them. |
| or_tokens_completion | OpenRouter completion tokens | model | tokens | W | sum | False | Output tokens, where the export separates them. |
| or_app_tokens | OpenRouter tokens by app x model | model_x_app | tokens | W | sum | False | Tokens a named app sent to a model. Entity id is '<app_id>\|<model_id>'. |
| or_tokens_free | OpenRouter tokens on free variant | model | tokens | D | sum | False | Subset of or_tokens served on the ':free' variant (zero price to the user). |
| or_requests | OpenRouter requests | model | requests | D | sum | False | Number of requests (all variants). |
| or_tokens_topweekly | OpenRouter tokens, weekly top-N chart | model | tokens | W | sum | False | Weekly tokens from the rankings chart; only the ~9 largest models each week appear (others are missing, not zero). Longer history than or_tokens. |
| or_tokens_platform | OpenRouter total tokens (all models) | market | tokens | W | sum | False | Sum of all chart series including 'Others' = all tokens routed that week. |
| or_token_share_platform | Share of all OpenRouter tokens (top-N weeks) | model | share [0,1] | W | mean | True | or_tokens_topweekly / or_tokens_platform. |
| or_app_tokens_day | App tokens, last day (top 20 apps) | app | tokens | snapshot | last | False | Tokens sent by the app in the last day, from the rankings page. |
| or_app_tokens_week | App tokens, last week (top 20 apps) | app | tokens | snapshot | last | False | Tokens sent by the app in the last 7 days. |
| or_app_tokens_month | App tokens, last month (top 20 apps) | app | tokens | snapshot | last | False | Tokens sent by the app in the last 30 days. |
| or_token_share | OpenRouter token share | model | share [0,1] | W | mean | True | or_tokens / sum of or_tokens over all models in the period (all models with daily activity data). |
| or_n_apps | Number of ranked apps using model | model | count | W | max | True | Distinct apps with positive or_app_tokens for the model in the period. |
| or_top_app_share | Share of model tokens from its largest app | model | share [0,1] | W | mean | True | Concentration check: max over apps of or_app_tokens / or_tokens. |

## github (tier B)

| name | label | entity_level | unit | native_freq | agg | derived | definition |
|---|---|---|---|---|---|---|---|
| gh_new_repos | New public repos mentioning provider | provider | repos | M | sum | False | GitHub repository-search total_count for the provider query restricted to repos created in the month. |
| gh_new_repos_all | All new public repos (normaliser) | market | repos | M | sum | False | total_count of all repositories created in the month; denominator for gh_new_repos_share. |
| gh_new_repos_share | Provider share of new repos | provider | share | M | mean | True | gh_new_repos / gh_new_repos_all. |

## pypi (tier B)

| name | label | entity_level | unit | native_freq | agg | derived | definition |
|---|---|---|---|---|---|---|---|
| pypi_downloads | PyPI downloads (no mirrors) | package | downloads | D | sum | False | Daily downloads of the SDK package excluding mirrors. |
| sdk_downloads | SDK downloads (PyPI + npm, provider) | provider | downloads | D | sum | True | Sum of first-party SDK package downloads mapped to the provider. |

## npm (tier B)

| name | label | entity_level | unit | native_freq | agg | derived | definition |
|---|---|---|---|---|---|---|---|
| npm_downloads | npm downloads | package | downloads | D | sum | False | Daily downloads of the SDK package from the npm registry. |

## huggingface (tier A)

| name | label | entity_level | unit | native_freq | agg | derived | definition |
|---|---|---|---|---|---|---|---|
| hf_downloads_30d | HF downloads, rolling 30 days | model | downloads | snapshot | last | False | Value of `downloads` at snapshot time (rolling 30-day window ending that day). |
| hf_downloads_alltime | HF downloads, all-time | model | downloads | snapshot | last | False | Cumulative downloads at snapshot time. |
| hf_likes | HF likes | model | likes | snapshot | last | False | Cumulative likes at snapshot time. |
| hf_n_derivatives | HF derivative models (lower bound) | model | repos | snapshot | max | False | Count of repos tagged base_model:<repo>; capped by page limit so a lower bound. |
| hf_downloads_period | HF downloads in period (derived) | model | downloads | snapshot | sum | True | First difference of hf_downloads_alltime between consecutive period-end snapshots. |

## provider_disclosures (tier B)

| name | label | entity_level | unit | native_freq | agg | derived | definition |
|---|---|---|---|---|---|---|---|
| disc_tokens_per_day | Disclosed tokens processed per day | provider | tokens/day | irregular | last | False | Tokens per day as stated; convert monthly figures by /days only when the source states the period. |
| disc_tokens_per_month | Disclosed tokens processed per month | provider | tokens/month | irregular | last | False | Tokens per month as stated. |
| disc_revenue_runrate_usd | Disclosed AI revenue run-rate | provider | USD/year | irregular | last | False | Annualised revenue run-rate as stated (not GAAP). |
| disc_revenue_usd | Disclosed AI/cloud revenue for period | provider | USD | Q | last | False | Recognised revenue for the stated period (segment where available). |
| disc_api_revenue_share | API share of revenue | provider | share | irregular | last | False | API/developer revenue as share of total, as stated. |
| disc_customers | Disclosed business customers | provider | count | irregular | last | False | Paying business/API customers as stated. |
| disc_developers | Disclosed developers | provider | count | irregular | last | False | Developers/API users as stated (definition varies; keep quote). |
| disc_weekly_users | Disclosed weekly active users (consumer) | provider | users | irregular | last | False | Consumer WAU as stated; context only, not API demand. |

## china_demand (tier C)

| name | label | entity_level | unit | native_freq | agg | derived | definition |
|---|---|---|---|---|---|---|---|
| cn_tokens_per_day | Announced daily token calls (CN) | provider | tokens/day | irregular | last | False | Daily token volume announced by a Chinese provider or cloud platform. |
| cn_platform_rank | Platform ranking position (CN) | model | rank (1=top) | irregular | mean | False | Ordinal position on a named Chinese ranking (store ranking name in note). |
| cn_enterprise_customers | Announced enterprise customers (CN) | provider | count | irregular | last | False | Enterprise customers announced. |

## external_benchmarks (tier C)

| name | label | entity_level | unit | native_freq | agg | derived | definition |
|---|---|---|---|---|---|---|---|
| ext_firm_adoption_rate | Share of firms using AI (official survey) | market | share | irregular | last | False | Statistical-office survey share of firms using AI; store country/wave. |
| ext_job_postings | Job postings mentioning provider/model | provider | postings | M | sum | False | Postings that mention the provider or SDK, from a named job-board dataset. |

## Panel keys and quality columns

| column | meaning |
|---|---|
| model_id | canonical model id (data/reference/model_crosswalk.csv) |
| provider_id | canonical provider id |
| app_id | OpenRouter app slug (app panel) |
| period | pandas Period at panel frequency (W or M) |
| n_obs | raw observations behind the cell (long panel) |
| sources | contributing source_ids |
| tier | best (lowest) tier among contributing sources |
| any_estimate | any contributing value was second-hand, rounded or a lower bound |
| part_a_price_date | date of the Part A price matched as-of period end |
| <price>_lag1 | Part A value in the previous period |
