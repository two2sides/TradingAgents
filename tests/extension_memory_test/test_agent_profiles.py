"""Tests for agent_profiles — role-specific memory retrieval config."""

from __future__ import annotations

import pytest

from tradingagents.extensions.memory.agent_profiles import (
    PROFILES,
    AgentMemoryProfile,
    _compute_basic_features,
    get_profile,
)
from .conftest import make_market, make_memory_query, NOW


# ── Profile registry ───────────────────────────────────────────────────

class TestProfileRegistry:
    def test_all_registered_roles_are_valid(self):
        """Every registered profile must be an AgentMemoryProfile."""
        for role, profile in PROFILES.items():
            assert isinstance(profile, AgentMemoryProfile), f"{role} is not an AgentMemoryProfile"
            assert profile.role == role

    def test_weights_sum_to_approximately_one(self):
        """Sim + recency + outcome weights should sum to ~1.0 for each profile."""
        for role, profile in PROFILES.items():
            total = profile.sim_weight + profile.recency_weight + profile.outcome_weight
            assert abs(total - 1.0) < 0.02, (
                f"{role} weights sum to {total}, expected ~1.0"
            )

    def test_get_profile_returns_correct_role(self):
        pm = get_profile("portfolio_manager")
        assert pm.role == "portfolio_manager"
        ma = get_profile("market_analyst")
        assert ma.role == "market_analyst"

    def test_get_profile_unknown_role_falls_back_to_pm(self):
        result = get_profile("nonexistent_role")
        assert result is PROFILES["portfolio_manager"]

    def test_portfolio_manager_sees_all_tags(self):
        pm = get_profile("portfolio_manager")
        assert pm.interest_tags == ["*"]
        assert pm.cross_ticker is True

    def test_bear_researcher_cares_most_about_outcomes(self):
        br = get_profile("bear_researcher")
        # Bear researcher should weight outcomes more than similarity
        assert br.outcome_weight > br.sim_weight

    def test_risk_roles_stay_on_same_ticker(self):
        """Risk-aggressive and conservative should not cross tickers."""
        for role in ("risk_aggressive", "risk_conservative"):
            p = get_profile(role)
            assert p.cross_ticker is False, f"{role} should stay on same ticker"

    def test_each_profile_produces_non_empty_query(self):
        """build_query must return a non-empty string for every role."""
        query = make_memory_query(symbol="NVDA")
        for role, profile in PROFILES.items():
            q = profile.build_query(query)
            assert isinstance(q, str) and len(q) > 20, f"{role} query too short: {q!r}"

    def test_query_includes_symbol(self):
        query = make_memory_query(symbol="NVDA")
        for role in ("portfolio_manager", "market_analyst", "fundamentals_analyst"):
            profile = get_profile(role)
            q = profile.build_query(query)
            assert "NVDA" in q, f"{role} query missing symbol"


# ── Feature extraction ─────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_empty_bars_returns_empty_dict(self):
        features = _compute_basic_features([])
        assert features == {}

    def test_single_bar_returns_empty_dict(self):
        market = make_market(n_bars=1)
        features = _compute_basic_features(market.bars)
        assert features == {}

    def test_enough_bars_extracts_expected_keys(self):
        market = make_market(n_bars=20)
        features = _compute_basic_features(market.bars)
        expected = {"return_5d", "return_5d_signed", "volatility", "volume_ratio",
                    "trend_direction", "price_position", "avg_spread"}
        assert set(features.keys()) == expected

    def test_trend_is_bullish_for_uptrend(self):
        market = make_market(n_bars=20)
        features = _compute_basic_features(market.bars)
        # Bars have monotonically increasing closes (base + i*0.5, close = open + 1.2)
        assert features["trend_direction"] == "bullish"

    def test_price_position_in_unit_range(self):
        market = make_market(n_bars=20)
        features = _compute_basic_features(market.bars)
        assert 0 <= features["price_position"] <= 1.0


# ── AgentMemoryProfile dataclass ────────────────────────────────────────

class TestAgentMemoryProfile:
    def test_default_factory_fields_are_independent(self):
        a = AgentMemoryProfile(role="a")
        b = AgentMemoryProfile(role="b")
        a.interest_tags.append("foo")
        assert "foo" not in b.interest_tags

    def test_to_retrieval_kwargs_returns_expected_keys(self):
        query = make_memory_query(symbol="MSFT")
        profile = get_profile("market_analyst")
        kwargs = profile.to_retrieval_kwargs(query)
        expected = {
            "query_text", "symbol", "cross_ticker", "chunk_types",
            "interest_tags", "sim_weight", "recency_weight", "outcome_weight",
            "max_items", "as_of", "limit",
        }
        assert set(kwargs.keys()) == expected
        assert kwargs["symbol"] == "MSFT"
        assert kwargs["max_items"] == profile.max_items
