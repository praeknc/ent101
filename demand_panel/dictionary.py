"""Variable dictionary for the demand-side panel.

Every value that enters the long observation table must use a `name` defined
here. `agg` says how to collapse native frequency to panel frequency:

  sum   flows (tokens, downloads, new repos) -> add up within the period
  last  stocks and snapshots (all-time downloads, customer counts) -> last value in period
  mean  rates and ranks -> average within the period
  max   lower-bound counts -> largest within the period

Variables with `derived=True` are computed in panel.py, never collected.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    source_id: str
    entity_level: str
    unit: str
    native_freq: str
    agg: str
    definition: str
    derived: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


VARIABLES: dict[str, Variable] = {v.name: v for v in [
    # ---------------- OpenRouter
    Variable("or_tokens", "OpenRouter tokens (total)", "openrouter", "model", "tokens", "W", "sum",
             "Prompt + completion tokens routed to the model in the period, summed across apps."),
    Variable("or_tokens_prompt", "OpenRouter prompt tokens", "openrouter", "model", "tokens", "W", "sum",
             "Input tokens, where the export separates them."),
    Variable("or_tokens_completion", "OpenRouter completion tokens", "openrouter", "model", "tokens", "W", "sum",
             "Output tokens, where the export separates them."),
    Variable("or_app_tokens", "OpenRouter tokens by app x model", "openrouter", "model_x_app", "tokens", "W", "sum",
             "Tokens a named app sent to a model. Entity id is '<app_id>|<model_id>'."),
    Variable("or_token_share", "OpenRouter token share", "openrouter", "model", "share [0,1]", "W", "mean",
             "or_tokens / sum of or_tokens over all models in the period (models in the ranking only).",
             derived=True),
    Variable("or_n_apps", "Number of ranked apps using model", "openrouter", "model", "count", "W", "max",
             "Distinct apps with positive or_app_tokens for the model in the period.", derived=True),
    Variable("or_top_app_share", "Share of model tokens from its largest app", "openrouter", "model",
             "share [0,1]", "W", "mean",
             "Concentration check: max over apps of or_app_tokens / or_tokens.", derived=True),
    # ---------------- GitHub
    Variable("gh_new_repos", "New public repos mentioning provider", "github", "provider", "repos", "M", "sum",
             "GitHub repository-search total_count for the provider query restricted to repos created in the month."),
    Variable("gh_new_repos_all", "All new public repos (normaliser)", "github", "market", "repos", "M", "sum",
             "total_count of all repositories created in the month; denominator for gh_new_repos_share."),
    Variable("gh_new_repos_share", "Provider share of new repos", "github", "provider", "share", "M", "mean",
             "gh_new_repos / gh_new_repos_all.", derived=True),
    # ---------------- package downloads
    Variable("pypi_downloads", "PyPI downloads (no mirrors)", "pypi", "package", "downloads", "D", "sum",
             "Daily downloads of the SDK package excluding mirrors."),
    Variable("npm_downloads", "npm downloads", "npm", "package", "downloads", "D", "sum",
             "Daily downloads of the SDK package from the npm registry."),
    Variable("sdk_downloads", "SDK downloads (PyPI + npm, provider)", "pypi", "provider", "downloads", "D", "sum",
             "Sum of first-party SDK package downloads mapped to the provider.", derived=True),
    # ---------------- Hugging Face
    Variable("hf_downloads_30d", "HF downloads, rolling 30 days", "huggingface", "model", "downloads",
             "snapshot", "last", "Value of `downloads` at snapshot time (rolling 30-day window ending that day)."),
    Variable("hf_downloads_alltime", "HF downloads, all-time", "huggingface", "model", "downloads",
             "snapshot", "last", "Cumulative downloads at snapshot time."),
    Variable("hf_likes", "HF likes", "huggingface", "model", "likes", "snapshot", "last",
             "Cumulative likes at snapshot time."),
    Variable("hf_n_derivatives", "HF derivative models (lower bound)", "huggingface", "model", "repos",
             "snapshot", "max", "Count of repos tagged base_model:<repo>; capped by page limit so a lower bound."),
    Variable("hf_downloads_period", "HF downloads in period (derived)", "huggingface", "model", "downloads",
             "snapshot", "sum", "First difference of hf_downloads_alltime between consecutive period-end snapshots.",
             derived=True),
    # ---------------- provider disclosures
    Variable("disc_tokens_per_day", "Disclosed tokens processed per day", "provider_disclosures", "provider",
             "tokens/day", "irregular", "last", "Tokens per day as stated; convert monthly figures by /days only "
             "when the source states the period."),
    Variable("disc_tokens_per_month", "Disclosed tokens processed per month", "provider_disclosures", "provider",
             "tokens/month", "irregular", "last", "Tokens per month as stated."),
    Variable("disc_revenue_runrate_usd", "Disclosed AI revenue run-rate", "provider_disclosures", "provider",
             "USD/year", "irregular", "last", "Annualised revenue run-rate as stated (not GAAP)."),
    Variable("disc_revenue_usd", "Disclosed AI/cloud revenue for period", "provider_disclosures", "provider",
             "USD", "Q", "last", "Recognised revenue for the stated period (segment where available)."),
    Variable("disc_api_revenue_share", "API share of revenue", "provider_disclosures", "provider", "share",
             "irregular", "last", "API/developer revenue as share of total, as stated."),
    Variable("disc_customers", "Disclosed business customers", "provider_disclosures", "provider", "count",
             "irregular", "last", "Paying business/API customers as stated."),
    Variable("disc_developers", "Disclosed developers", "provider_disclosures", "provider", "count",
             "irregular", "last", "Developers/API users as stated (definition varies; keep quote)."),
    Variable("disc_weekly_users", "Disclosed weekly active users (consumer)", "provider_disclosures", "provider",
             "users", "irregular", "last", "Consumer WAU as stated; context only, not API demand."),
    # ---------------- China
    Variable("cn_tokens_per_day", "Announced daily token calls (CN)", "china_demand", "provider", "tokens/day",
             "irregular", "last", "Daily token volume announced by a Chinese provider or cloud platform."),
    Variable("cn_platform_rank", "Platform ranking position (CN)", "china_demand", "model", "rank (1=top)",
             "irregular", "mean", "Ordinal position on a named Chinese ranking (store ranking name in note)."),
    Variable("cn_enterprise_customers", "Announced enterprise customers (CN)", "china_demand", "provider", "count",
             "irregular", "last", "Enterprise customers announced."),
    # ---------------- external benchmarks (validation only)
    Variable("ext_firm_adoption_rate", "Share of firms using AI (official survey)", "external_benchmarks", "market",
             "share", "irregular", "last", "Statistical-office survey share of firms using AI; store country/wave."),
    Variable("ext_job_postings", "Job postings mentioning provider/model", "external_benchmarks", "provider",
             "postings", "M", "sum", "Postings that mention the provider or SDK, from a named job-board dataset."),
]}


def get(name: str) -> Variable:
    try:
        return VARIABLES[name]
    except KeyError as e:
        raise KeyError(f"Variable '{name}' is not in the dictionary. Add it to demand_panel/dictionary.py.") from e


def agg_rule(name: str) -> str:
    return get(name).agg
