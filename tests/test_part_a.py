"""Part A cleaning rules on small synthetic inputs (no workbook needed)."""
import numpy as np
import pandas as pd
import pytest

from demand_panel import part_a as pa


def raw(rows):
    base = dict(host="openai", section="Standard", event="snapshot", price_cached_in=np.nan,
                currency="USD", unit="per 1M", flags="", source_url="", origin_file="t")
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["price_in_local"] = df.get("price_in_local", df["price_in"]).fillna(df["price_in"])
    df["price_out_local"] = df.get("price_out_local", df["price_out"]).fillna(df["price_out"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["host", "published", "key", "section", "event", "currency", "unit", "flags", "source"]:
        df[c] = df[c].astype(str)
    return df


def run(rows, **kw):
    res = pa.select_observations(pa.classify(raw(rows)))
    return res, pa.monthly_panel(res, **kw)


def test_model_key_normaliser():
    assert pa.model_key("Claude 3.5 Sonnet") == "claude-sonnet-3-5"
    assert pa.model_key("claude-3-5-sonnet-20241022") == "claude-sonnet-3-5"
    assert pa.model_key("dashscope/qwen-plus-2025-07-14 [thinking]") == "qwen-plus"
    assert pa.model_key("gpt-5.5 (<272K context length)") == "gpt-5-5"
    assert pa.model_key("gpt-4o-2024-08-06", strip_date=False) == "gpt-4o-2024-08-06"


def test_unlabelled_tier_resolved_against_labelled_price():
    rows = [dict(source="L", published="gpt-x", key="gpt-x", date="2026-01-10", section="chat",
                 event="first_seen", price_in=2.5, price_out=10)]
    for pin, pout in [(4.25, 17), (2.5, 10), (1.25, 5)]:
        rows.append(dict(source="W", published="gpt-x", key="gpt-x", date="2026-03-01",
                         section="tier not labelled on archived page",
                         flags="tier unknown: Standard, Batch, Flex, Priority or fine-tuned", price_in=pin,
                         price_out=pout))
    res, p = run(rows)
    w = res.obs[res.obs["source"] == "W"]
    assert len(w) == 1 and w["price_in"].iloc[0] == 2.5
    assert w["tier_resolution"].iloc[0] == "matched_to_labelled"


def test_unlabelled_tier_without_reference_is_dropped():
    rows = [dict(source="W", published="gpt-y", key="gpt-y", date="2026-03-01",
                 section="tier not labelled on archived page", flags="tier unknown", price_in=x, price_out=4 * x)
            for x in (1.0, 2.0, 0.5)]
    res, p = run(rows)
    assert res.obs.empty or (res.obs["key"] != "gpt-y").all()
    assert (res.rows["drop_reason"] == "tier_unresolved").sum() == 3


def test_dated_variant_and_fast_and_us_prefixes():
    rows = [
        dict(source="L", published="claude-z", key="claude-z", date="2026-02-05", section="chat",
             event="first_seen", price_in=5, price_out=25, host="anthropic"),
        dict(source="L", published="claude-z-20260205", key="claude-z", date="2026-02-05", section="chat",
             event="first_seen", price_in=9, price_out=9, host="anthropic"),
        dict(source="L", published="us/claude-z", key="claude-z", date="2026-02-06", section="chat",
             event="first_seen", price_in=5.5, price_out=27.5, host="anthropic"),
        dict(source="L", published="fast/claude-z", key="claude-z", date="2026-02-09", section="chat",
             event="first_seen", price_in=30, price_out=150, host="anthropic"),
    ]
    res, p = run(rows, end="2026-03")
    assert set(p["price_in_usd_per_mtok"]) == {5.0}
    dr = res.rows.set_index("published")["drop_reason"]
    assert dr["fast/claude-z"] == "non_standard_tier"
    assert dr["us/claude-z"] == "non_preferred_region"
    assert dr["claude-z-20260205"] == "dated_variant_of_key"


def test_litellm_cache_change_does_not_undo_archived_cut():
    rows = [
        dict(source="L", published="gem", key="gem", host="google", date="2024-05-01", section="chat",
             event="first_seen", price_in=0.35, price_out=1.05),
        dict(source="W", published="gem", key="gem", host="google", date="2024-08-15", price_in=0.075,
             price_out=0.3),
        dict(source="L", published="gem", key="gem", host="google", date="2024-10-01", section="chat",
             event="cache_changed", price_in=0.35, price_out=1.05, price_cached_in=0.01),
    ]
    _, p = run(rows, end="2024-11")
    oct_ = p[p["period"] == "2024-10"].iloc[0]
    assert oct_["price_in_usd_per_mtok"] == 0.075
    assert oct_["price_cached_in_usd_per_mtok"] == 0.01
    assert p[p["period"] == "2024-08"]["price_cut"].iloc[0] == 1


def test_carry_rules_snapshot_capped_litellm_until_removal():
    rows = [
        dict(source="W", published="snap", key="snap", date="2024-01-15", price_in=1, price_out=2),
        dict(source="L", published="log", key="log", date="2024-01-15", section="chat", event="first_seen",
             price_in=1, price_out=2),
        dict(source="L", published="log", key="log", date="2024-12-20", section="chat", event="removed",
             price_in=1, price_out=2),
    ]
    _, p = run(rows, max_carry_months=3, end="2025-06")
    assert p[p.model_id == "snap"]["period"].max() == "2024-04"
    assert p[p.model_id == "log"]["period"].max() == "2024-11"


def test_cny_price_change_measured_in_local_currency():
    rows = [dict(source="W", published="ernie-a", key="ernie-a", host="baidu", currency="CNY", section="文心",
                 date=d, price_in=usd, price_out=usd, price_in_local=120, price_out_local=120)
            for d, usd in [("2024-03-10", 16.66), ("2024-04-10", 16.58)]]
    _, p = run(rows, end="2024-04")
    assert p["market"].unique().tolist() == ["CN"]
    assert p.iloc[-1]["price_change_pct"] == 0.0


def test_non_text_and_long_context_dropped():
    rows = [
        dict(source="Live", published="gpt-image-9", key="gpt-image-9", date="2026-11-17", price_in=5, price_out=40),
        dict(source="Live", published="qwen-q [non-thinking] (256K<Token≤1M)", key="qwen-q", host="alibaba",
             section="Qwen / Singapore / International", date="2026-11-17", price_in=1.2, price_out=3.6,
             flags="higher input-length tier"),
        dict(source="Live", published="qwen-q [non-thinking]", key="qwen-q", host="alibaba",
             section="Qwen / Singapore / International", date="2026-11-17", price_in=0.4, price_out=1.2),
        dict(source="Live", published="qwen-q [thinking]", key="qwen-q", host="alibaba",
             section="Qwen / Singapore / International", date="2026-11-17", price_in=0.4, price_out=4.0),
        dict(source="Live", published="qwen-q [non-thinking]", key="qwen-q", host="alibaba",
             section="Qwen / United States (Virginia) / Global", date="2026-11-17", price_in=0.115, price_out=0.287),
    ]
    _, p = run(rows)
    assert list(p["model_id"]) == ["qwen-q"]
    assert p.iloc[0]["price_out_usd_per_mtok"] == 1.2


def test_price_index_chained_jevons():
    panel = pd.DataFrame({
        "model_id": ["a", "a", "b", "b"], "period": ["2025-01", "2025-02"] * 2,
        "date": pd.to_datetime(["2025-01-31", "2025-02-28"] * 2),
        "price_blended_3to1_local": [1.0, 0.5, 2.0, 2.0], "currency_local": "USD", "maker": "x"})
    idx = pa.price_index(panel)
    assert idx["price_index_all"].iloc[0] == 100
    assert idx["price_index_all"].iloc[1] == pytest.approx(100 * np.sqrt(0.5))
