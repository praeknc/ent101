"""Source registry for the demand-side panel (thesis section 4.4).

Each source gets:
  * a tier (A/B/C) that says how much weight its numbers can bear,
  * what it CAN and CANNOT show,
  * sampling-bias notes,
  * the entity level it observes and the level it merges to Part A at.

Tier definitions
----------------
A  Systematic, machine-readable log of real behaviour, with a consistent
   definition over time and a known population (even if the population
   is self-selected). Usable as a dependent variable in regressions.
B  Behavioural but indirect proxy, or first-party aggregate numbers that are
   attributable but unaudited and irregular. Usable as a dependent variable
   only with care (proxy error, sparse timing); good for robustness.
C  Announcements, rankings, survey results or third-party estimates.
   Definitions shift between releases; use for description, triangulation
   and validation, not as a main outcome.

`validation_only=True` means the source must never enter the estimation panel
as an outcome; it is kept to check that the main measures move with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    category: str              # usage_aggregator | developer_activity | provider_disclosure | china | external_benchmark
    tier: str                  # A | B | C
    entity_level: str          # model | app | provider | package | model_x_app | market
    native_freq: str           # D | W | M | Q | irregular | snapshot
    region: str                # global | CN | US | ...
    collection: str            # api | export | manual
    can_show: str
    cannot_show: str
    sampling_bias: List[str] = field(default_factory=list)
    merge_level: str = "model"  # level at which it joins Part A
    validation_only: bool = False
    url: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sampling_bias"] = " | ".join(self.sampling_bias)
        return d


SOURCES: dict[str, Source] = {s.source_id: s for s in [
    # ------------------------------------------------------------------ usage aggregators
    Source(
        source_id="openrouter",
        name="OpenRouter rankings (token volume by model and by app)",
        category="usage_aggregator", tier="A", entity_level="model_x_app",
        native_freq="W", region="global", collection="export",
        can_show="Relative token demand across models from many providers routed through one "
                 "gateway; switching between models over time; which apps drive which models; "
                 "response of usage share to price changes on the same platform.",
        cannot_show="Total market demand or first-party API traffic (direct OpenAI/Anthropic/"
                    "Google API calls and consumer chat apps never touch the router); revenue; "
                    "enterprise contracts; usage inside China.",
        sampling_bias=[
            "Self-selected users: price-sensitive developers, hobbyists and agent/coding tools "
            "that want multi-model routing are over-represented; large enterprises are under-represented.",
            "A few heavy apps (coding agents, role-play) dominate tokens; one app switching model "
            "can move a model's share by several points. Report shares with and without the top apps.",
            "Free and promotional model variants inflate tokens for some open-weight models; "
            "keep ':free' variants as a separate flag rather than merging them silently.",
            "Rankings show only the top-N models/apps per period; long tail is left-censored "
            "(record as missing, not zero).",
            "Platform growth itself raises all levels; use shares or normalise by platform total.",
        ],
        merge_level="model",
        url="https://openrouter.ai/rankings",
    ),
    # ------------------------------------------------------------------ developer activity
    Source(
        source_id="github",
        name="GitHub new repositories mentioning a provider SDK or endpoint",
        category="developer_activity", tier="B", entity_level="provider",
        native_freq="M", region="global", collection="api",
        can_show="Flow of new developer projects building on each provider; timing of "
                 "developer interest after launches and price cuts.",
        cannot_show="Production usage or spend; private/enterprise repositories; whether a "
                    "mentioned SDK is actually called; model-level choice within a provider.",
        sampling_bias=[
            "Public repositories only; hobby, tutorial and student projects dominate counts.",
            "Keyword search matches mentions, not imports; tune queries and audit a random sample "
            "(record precision in the query table).",
            "OpenAI-compatible endpoints mean many non-OpenAI projects mention 'openai'; "
            "this inflates OpenAI counts.",
            "GitHub user base grows over time; normalise by all new repos in the same month.",
            "Search index is revised; counts for a past month can change when re-queried. Keep retrieval date.",
        ],
        merge_level="provider",
        url="https://docs.github.com/en/rest/search/search",
    ),
    Source(
        source_id="pypi",
        name="PyPI downloads of provider SDK packages",
        category="developer_activity", tier="B", entity_level="package",
        native_freq="D", region="global", collection="api",
        can_show="Installation activity for first-party SDKs (openai, anthropic, google-genai, "
                 "mistralai, ...); trend and relative growth across providers.",
        cannot_show="Distinct users or projects; spend; model-level demand; usage through "
                    "raw HTTP or OpenAI-compatible SDKs pointed at other providers.",
        sampling_bias=[
            "Dominated by CI pipelines and container builds, not people; levels are not users.",
            "The 'openai' package is used to call many other providers via compatible endpoints.",
            "Mirror traffic excluded (pypistats without_mirrors); version bumps cause re-install spikes.",
            "pypistats API keeps only ~180 days; longer history needs the BigQuery public dataset.",
        ],
        merge_level="provider",
        url="https://pypistats.org/api/",
    ),
    Source(
        source_id="npm",
        name="npm downloads of provider SDK packages",
        category="developer_activity", tier="B", entity_level="package",
        native_freq="D", region="global", collection="api",
        can_show="JavaScript/TypeScript SDK installation activity; complements PyPI for web apps.",
        cannot_show="Same limits as PyPI: not users, not spend, not model choice.",
        sampling_bias=[
            "CI and bundler installs dominate; dependency of popular frameworks drives counts.",
            "Scoped packages (@anthropic-ai/sdk, @google/genai) were renamed over time; "
            "chain old and new package names in the package map.",
        ],
        merge_level="provider",
        url="https://github.com/npm/registry/blob/main/docs/download-counts.md",
    ),
    Source(
        source_id="huggingface",
        name="Hugging Face downloads and derivatives of open-weight models",
        category="developer_activity", tier="A", entity_level="model",
        native_freq="snapshot", region="global", collection="api",
        can_show="Demand for self-hosted open weights (downloads, likes, number of fine-tunes); "
                 "the outside option to paid APIs; Chinese open-weight models' global uptake.",
        cannot_show="Inference volume after download; closed models; downloads via mirrors "
                    "(ModelScope, hf-mirror) or other hosts; historical values before collection "
                    "started (API returns only rolling 30-day and all-time totals).",
        sampling_bias=[
            "One download = one file fetch by any client, incl. CI and re-downloads; not users.",
            "Quantised re-uploads by third parties split demand across repos; aggregate via the "
            "crosswalk's base model, keep repo-level too.",
            "No history from the API: a panel exists only from the first snapshot. Schedule "
            "collection weekly; derive period flows from all-time differences.",
            "Chinese users often download from ModelScope; HF under-counts CN demand.",
        ],
        merge_level="model",
        url="https://huggingface.co/docs/hub/api",
    ),
    # ------------------------------------------------------------------ provider disclosures
    Source(
        source_id="provider_disclosures",
        name="Provider disclosures: filings, earnings calls, official blog/keynote statements",
        category="provider_disclosure", tier="B", entity_level="provider",
        native_freq="irregular", region="global", collection="manual",
        can_show="Absolute scale: tokens processed, AI revenue run-rate, API vs. consumer split, "
                 "customer and developer counts. Anchors the relative measures to levels.",
        cannot_show="Model-level or price-tier detail; consistent definitions over time; "
                    "anything a firm chooses not to disclose (selection on good news).",
        sampling_bias=[
            "Unaudited and strategically timed: firms disclose when numbers look good.",
            "Definitions drift (monthly vs. daily tokens, run-rate vs. recognised revenue, "
            "'developers' vs. 'customers'). Store the exact quote and unit; never harmonise silently.",
            "Sparse: a handful of points per firm per year. Do not interpolate into the monthly panel; "
            "use as level anchors or in a separate provider x quarter panel.",
            "Private firms (OpenAI, Anthropic) disclose via press and interviews, not filings: "
            "mark is_estimate when the figure is reported second-hand.",
        ],
        merge_level="provider",
    ),
    # ------------------------------------------------------------------ China
    Source(
        source_id="china_demand",
        name="Chinese demand: public usage announcements and platform rankings",
        category="china", tier="C", entity_level="provider",
        native_freq="irregular", region="CN", collection="manual",
        can_show="Order-of-magnitude token volumes announced by Chinese providers and cloud "
                 "platforms (e.g. daily token calls at developer conferences); app-store and "
                 "platform ranking positions; direction of change around domestic price cuts.",
        cannot_show="A comparable time series; model-level usage shares; any audited measure; "
                    "usage of foreign models in China (largely unavailable).",
        sampling_bias=[
            "Announcement-driven: numbers appear at launches and conferences, gaps in between.",
            "Metric definitions differ by firm (daily vs. cumulative tokens; calls vs. tokens).",
            "Rankings are ordinal and published by parties with commercial interests.",
            "Expect THIN coverage. The coverage report lists every model x month cell with no "
            "Chinese demand data; thesis text must state these gaps rather than impute them.",
        ],
        merge_level="provider",
    ),
    # ------------------------------------------------------------------ external benchmarks
    Source(
        source_id="external_benchmarks",
        name="Official adoption surveys and job postings",
        category="external_benchmark", tier="C", entity_level="market",
        native_freq="irregular", region="global", collection="manual",
        can_show="Economy-wide adoption rates (e.g. statistical-office business surveys) and "
                 "labour demand for provider-specific skills; used to check that platform-based "
                 "measures track broader adoption.",
        cannot_show="Model choice or price response; anything at weekly frequency.",
        sampling_bias=[
            "Survey frames, questions and waves differ by country and year.",
            "Job-posting keyword counts reflect hiring cycles and posting practices, not usage.",
        ],
        merge_level="market",
        validation_only=True,
    ),
]}


def get(source_id: str) -> Source:
    try:
        return SOURCES[source_id]
    except KeyError as e:
        raise KeyError(f"Unknown source_id '{source_id}'. Known: {sorted(SOURCES)}") from e


def tier_of(source_id: str) -> str:
    return get(source_id).tier
