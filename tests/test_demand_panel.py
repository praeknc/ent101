"""Tests on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from demand_panel import coverage, panel, schema
from demand_panel.collectors import manual, openrouter
from demand_panel.crosswalk import Crosswalk
from demand_panel.merge import merge_index, merge_prices, rollup_prices_to_provider


@pytest.fixture(scope="module")
def cw():
    return Crosswalk.load()


def test_crosswalk_resolves_aliases(cw):
    assert cw.model_id("deepseek/deepseek-chat") == "deepseek/deepseek-v3"
    assert cw.model_id("deepseek/deepseek-r1:free") == "deepseek/deepseek-r1"
    assert cw.model_id("meta-llama/Llama-3.1-8B-Instruct") == "meta/llama-3.1-8b-instruct"
    assert cw.model_id("gpt-4o") == "openai/gpt-4o"
    _, unmatched = cw.resolve(["openai/gpt-4o", "someone/unknown-model"])
    assert unmatched == ["someone/unknown-model"]


def test_schema_rejects_unknown_and_derived():
    base = dict(source_id="openrouter", entity_type="model", entity_id="x",
                period_start="2025-01-01", period_end="2025-01-07", value=1)
    with pytest.raises(ValueError):
        schema.make_obs([{**base, "variable": "not_a_var"}])
    with pytest.raises(ValueError):
        schema.make_obs([{**base, "variable": "or_token_share"}])
    ok = schema.make_obs([{**base, "variable": "or_tokens"}])
    assert ok.loc[0, "tier"] == "A" and ok.loc[0, "unit"] == "tokens"


@pytest.fixture
def or_export(tmp_path):
    rows = []
    # Week of 2025-01-27..02-02 straddles Jan/Feb: 5 days Jan, 2 days Feb
    for ws, a, b in [("2025-01-06", 700, 300), ("2025-01-27", 700, 700), ("2025-02-10", 100, 900)]:
        rows += [dict(period_start=ws, model_slug="openai/gpt-4o", app="", tokens=a),
                 dict(period_start=ws, model_slug="deepseek/deepseek-r1:free", app="", tokens=b),
                 dict(period_start=ws, model_slug="openai/gpt-4o", app="Cline", tokens=a * 0.6),
                 dict(period_start=ws, model_slug="openai/gpt-4o", app="SillyTavern", tokens=a * 0.4),
                 dict(period_start=ws, model_slug="deepseek/deepseek-r1:free", app="Cline", tokens=b)]
    rows.append(dict(period_start="2025-01-06", model_slug="mystery/model", app="", tokens=5))
    p = tmp_path / "or.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_openrouter_model_panel(cw, or_export):
    obs, unmatched = openrouter.load_export(or_export, cw)
    assert unmatched == ["mystery/model"]
    assert obs.loc[obs["entity_id"] == "deepseek/deepseek-r1:free", "note"].str.contains("free").all()
    with pytest.warns(UserWarning):
        wide, long = panel.build_model_panel(obs, "M")
    jan = wide[wide["period"] == pd.Period("2025-01", "M")].set_index("model_id")
    # gpt-4o Jan = 700 + 700*5/7 = 1200 ; r1 Jan = 300 + 700*5/7 = 800
    assert jan.loc["openai/gpt-4o", "or_tokens"] == pytest.approx(1200)
    assert jan.loc["deepseek/deepseek-r1", "or_tokens"] == pytest.approx(800)
    assert jan["or_token_share"].sum() == pytest.approx(1.0)
    assert jan.loc["openai/gpt-4o", "or_n_apps"] == 2
    assert jan.loc["openai/gpt-4o", "or_top_app_share"] == pytest.approx(0.6)
    # total tokens conserved by pro-rating
    assert wide["or_tokens"].sum() == pytest.approx(700 + 300 + 1400 + 1000)

    app = panel.build_app_panel(obs, "M")
    cline_jan = app[(app["app_id"] == "cline") & (app["period"] == pd.Period("2025-01", "M"))]
    assert cline_jan["share_in_app"].sum() == pytest.approx(1.0)


def _hf_obs(dates_vals, repo="deepseek-ai/DeepSeek-R1", model="deepseek/deepseek-r1"):
    rows = []
    for d, v in dates_vals:
        rows.append(dict(source_id="huggingface", entity_type="model", entity_id=repo, model_id=model,
                         provider_id="deepseek", period_start=d, period_end=d,
                         variable="hf_downloads_alltime", value=v))
    return schema.make_obs(rows)


def test_hf_period_flows_from_snapshots():
    obs = _hf_obs([("2025-01-10", 100), ("2025-01-31", 150), ("2025-02-28", 260), ("2025-04-30", 500)])
    wide, _ = panel.build_model_panel(obs, "M")
    w = wide.set_index("period")
    assert w.loc[pd.Period("2025-01", "M"), "hf_downloads_alltime"] == 150   # last in month
    assert w.loc[pd.Period("2025-02", "M"), "hf_downloads_period"] == 110
    # March missing -> April flow not attributable
    assert np.isnan(w.loc[pd.Period("2025-04", "M"), "hf_downloads_period"])


def test_provider_panel_sdk_and_github_share(cw):
    rows = []
    for d in pd.date_range("2025-01-01", "2025-02-28"):
        rows.append(dict(source_id="pypi", entity_type="package", entity_id="anthropic",
                         provider_id="anthropic", period_start=d, period_end=d,
                         variable="pypi_downloads", value=10))
        rows.append(dict(source_id="npm", entity_type="package", entity_id="@anthropic-ai/sdk",
                         provider_id="anthropic", period_start=d, period_end=d,
                         variable="npm_downloads", value=5))
    for m, n, tot in [("2025-01", 50, 1000), ("2025-02", 80, 1000)]:
        p = pd.Period(m, "M")
        rows.append(dict(source_id="github", entity_type="provider", entity_id="anthropic_sdk",
                         provider_id="anthropic", period_start=p.start_time, period_end=p.end_time.normalize(),
                         variable="gh_new_repos", value=n))
        rows.append(dict(source_id="github", entity_type="market", entity_id="all_new_repos",
                         period_start=p.start_time, period_end=p.end_time.normalize(),
                         variable="gh_new_repos_all", value=tot))
    rows.append(dict(source_id="external_benchmarks", entity_type="provider", entity_id="anthropic",
                     provider_id="anthropic", period_start="2025-01-01", period_end="2025-01-31",
                     variable="ext_job_postings", value=999))
    obs = schema.make_obs(rows)
    wide, _ = panel.build_provider_panel(obs, "M")
    jan = wide.set_index(["provider_id", "period"]).loc[("anthropic", pd.Period("2025-01", "M"))]
    assert jan["pypi_downloads"] == 310 and jan["npm_downloads"] == 155
    assert jan["sdk_downloads"] == 465
    assert jan["gh_new_repos_share"] == pytest.approx(0.05)
    assert "ext_job_postings" not in wide.columns  # validation-only excluded by default
    wide2, _ = panel.build_provider_panel(obs, "M", include_validation=True)
    assert "ext_job_postings" in wide2.columns


def test_manual_validation(tmp_path, cw):
    p = tmp_path / "disc.csv"
    good = dict(source_id="provider_disclosures", provider_id="alibaba", model_raw="", region="global",
                period_start="2025-06-30", period_end="2025-06-30", variable="disc_customers", value="1000",
                unit_as_stated="customers", is_estimate="False", source_url="https://example.com/filing",
                source_doc="20-F", page_or_timestamp="p. 12", quote="over 1,000 customers",
                date_published="2025-07-15", note="")
    pd.DataFrame([good]).to_csv(p, index=False)
    obs = manual.load(p, cw, "provider_disclosures")
    assert obs.loc[0, "tier"] == "B" and "quote=" in obs.loc[0, "note"]
    bad = {**good, "source_url": "", "variable": "cn_tokens_per_day"}
    pd.DataFrame([bad]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="source_url is required"):
        manual.load(p, cw, "provider_disclosures")


def test_merge_prices_asof_and_lag(cw, or_export):
    obs, _ = openrouter.load_export(or_export, cw)
    with pytest.warns(UserWarning):
        wide, _ = panel.build_model_panel(obs, "M")
    prices = pd.DataFrame({
        "model_id": ["openai/gpt-4o", "openai/gpt-4o", "deepseek/deepseek-r1", "not/in-demand"],
        "date": ["2024-12-01", "2025-02-15", "2025-01-20", "2025-01-01"],
        "price_blended_usd_per_mtok": [10.0, 5.0, 2.0, 1.0],
    })
    merged, diag = merge_prices(wide, prices, cw)
    g = merged.set_index(["model_id", "period"])
    assert g.loc[("openai/gpt-4o", pd.Period("2025-01", "M")), "price_blended_usd_per_mtok"] == 10.0
    assert g.loc[("openai/gpt-4o", pd.Period("2025-02", "M")), "price_blended_usd_per_mtok"] == 5.0
    assert g.loc[("openai/gpt-4o", pd.Period("2025-02", "M")), "price_blended_usd_per_mtok_lag1"] == 10.0
    assert diag["models_in_part_a_not_in_demand"] == ["not/in-demand"]
    assert diag["row_match_rate"] == 1.0

    idx = pd.DataFrame({"date": ["2025-01-01", "2025-02-01"], "price_index": [100.0, 80.0]})
    m2, d2 = merge_index(merged, idx)
    assert d2["row_match_rate"] == 1.0

    w = wide[["model_id", "period", "or_tokens"]].rename(columns={"or_tokens": "w"})
    pp = rollup_prices_to_provider(prices, cw, weights=w)
    assert {"provider_id", "period", "price_blended_usd_per_mtok"} <= set(pp.columns)


def test_china_gaps(cw):
    rows = [dict(source_id="china_demand", entity_type="provider", entity_id="alibaba",
                 provider_id="alibaba", region="CN", period_start="2025-03-01", period_end="2025-03-01",
                 variable="cn_tokens_per_day", value=1e12, is_estimate=True,
                 source_url="https://example.com", note="announcement")]
    obs = schema.make_obs(rows)
    g = coverage.china_gaps(obs, cw, "2025-01", "2025-06")
    ali = g[g["provider_id"] == "alibaba"]
    assert ali["gap"].sum() == 5
    assert set(g["provider_id"]) == {"alibaba", "deepseek"}


def test_store_roundtrip(tmp_path, cw, or_export):
    obs, _ = openrouter.load_export(or_export, cw)
    path = tmp_path / "obs.csv"
    schema.append_obs(obs, path)
    again = schema.append_obs(obs, path)  # idempotent
    assert len(again) == len(obs)
    assert len(schema.read_obs(path)) == len(obs)
