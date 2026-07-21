"""Role-specific memory retrieval profiles.

Each agent role sees the market through a different lens. This module
defines what each role cares about, how its query is constructed, and how
retrieved memories are weighted for relevance ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradingagents.extensions.contracts import MemoryQuery


def _compute_basic_features(bars: list) -> dict[str, float]:
    """Extract lightweight features from OHLCV bars for query construction.

    Returns a dict of feature_name → value that profiles can reference in
    their query templates.  All computations are deterministic and side-effect-free.
    """
    if not bars or len(bars) < 2:
        return {}

    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]

    features: dict[str, float] = {}

    # Recent return (last 5 bars or all available)
    window = min(5, len(closes) - 1)
    if window > 0:
        features["return_5d"] = (closes[-1] / closes[-window - 1] - 1) * 100

    # Volatility (std of daily returns, annualised approx)
    if len(closes) >= 5:
        daily_rets = [
            (closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))
        ]
        mean_ret = sum(daily_rets) / len(daily_rets)
        variance = sum((r - mean_ret) ** 2 for r in daily_rets) / len(daily_rets)
        features["volatility"] = (variance ** 0.5) * 100  # daily σ in %

    # Volume trend (last 5 vs prior)
    if len(volumes) >= 10:
        recent_vol = sum(volumes[-5:]) / 5
        prior_vol = sum(volumes[-10:-5]) / 5
        features["volume_ratio"] = recent_vol / prior_vol if prior_vol > 0 else 1.0

    # Trend direction
    if len(closes) >= 10:
        ma_short = sum(closes[-5:]) / 5
        ma_long = sum(closes[-10:]) / 10
        features["trend"] = 1.0 if ma_short > ma_long else -1.0

    # Price relative to recent range
    if len(closes) >= 20:
        high = max(closes[-20:])
        low = min(closes[-20:])
        features["price_position"] = (
            (closes[-1] - low) / (high - low) if high != low else 0.5
        )

    # Average bar range (high-low spread)
    spreads = [(b.high - b.low) / b.close for b in bars[-10:]]
    features["avg_spread"] = (sum(spreads) / len(spreads)) * 100

    return features


# ── Profile definitions ────────────────────────────────────────────────────


@dataclass
class AgentMemoryProfile:
    """What and how a specific agent role retrieves from memory."""

    role: str
    # Chunk types this role cares about (ordered by preference)
    chunk_types: list[str] = field(default_factory=lambda: ["thesis", "reflection"])
    # Metadata tags used for candidate filtering
    interest_tags: list[str] = field(default_factory=list)
    # Weights for the hybrid score: similarity + recency + outcome_quality = 1.0
    sim_weight: float = 0.5
    recency_weight: float = 0.3
    outcome_weight: float = 0.2
    # Whether to include cross-ticker memories
    cross_ticker: bool = True
    # Maximum items returned
    max_items: int = 3

    def build_query(self, query: MemoryQuery) -> str:
        """Construct a role-specific natural-language query from market state."""
        features = _compute_basic_features(query.market.bars)
        template = _QUERY_TEMPLATES.get(self.role, _QUERY_TEMPLATES["portfolio_manager"])
        return template.format(symbol=query.symbol, **features)

    def to_retrieval_kwargs(self, query: MemoryQuery) -> dict:
        """Convert profile + query into kwargs for the retrieval pipeline."""
        return {
            "query_text": self.build_query(query),
            "symbol": query.symbol,
            "cross_ticker": self.cross_ticker,
            "chunk_types": self.chunk_types,
            "interest_tags": self.interest_tags,
            "sim_weight": self.sim_weight,
            "recency_weight": self.recency_weight,
            "outcome_weight": self.outcome_weight,
            "max_items": self.max_items,
            "as_of": query.as_of,
            "limit": query.limit,
        }


# ── Query templates ─────────────────────────────────────────────────────────
# Each template is a format string that receives ``symbol`` and the features
# computed from the current market snapshot.

_QUERY_TEMPLATES: dict[str, str] = {
    "fundamentals_analyst": (
        "Company valuation and fundamentals analysis for {symbol}. "
        "Key considerations: PE ratio trends, revenue growth trajectory, "
        "profit margins, debt levels, free cash flow yield, earnings surprises. "
        "Past decisions made under similar fundamental conditions where "
        "the investment thesis was validated or invalidated by outcomes."
    ),
    "market_analyst": (
        "Technical analysis pattern for {symbol}. "
        "Recent price return {return_5d:+.1f}%, volatility {volatility:.1f}%, "
        "volume ratio {volume_ratio:.2f}x, trend direction {'bullish' if trend>0 else 'bearish'}, "
        "price at {price_position:.0%} of recent range, spread {avg_spread:.1f}%. "
        "Similar technical conditions where trading decisions were made "
        "and what the outcomes revealed about the pattern reliability."
    ),
    "sentiment_analyst": (
        "Market sentiment and news analysis for {symbol}. "
        "News sentiment direction, social media mood, macro narrative positioning. "
        "Past decisions where sentiment analysis played a key role in the thesis, "
        "and whether the sentiment signal proved accurate or misleading."
    ),
    "news_analyst": (
        "Macroeconomic and global news context relevant to {symbol}. "
        "Policy changes, geopolitical events, sector rotations, macro regime shifts. "
        "Past decisions influenced by macro narratives and how those narratives "
        "translated into actual market outcomes."
    ),
    "bull_researcher": (
        "Bullish investment thesis for {symbol}: growth catalysts, "
        "upside potential, undervaluation arguments, competitive advantages. "
        "Previous bullish theses — which held, which failed, and what "
        "differentiated the successful calls from the unsuccessful ones."
    ),
    "bear_researcher": (
        "Bearish risk analysis for {symbol}: downside risks, "
        "overvaluation concerns, competitive threats, structural headwinds. "
        "Previous bearish theses — which were validated, which were premature, "
        "and what risk factors actually materialized."
    ),
    "risk_aggressive": (
        "Aggressive risk posture for {symbol}: reward asymmetry, "
        "growth momentum, positive gamma scenarios, high-conviction entry points. "
        "Past aggressive stances — when did leaning in pay off, "
        "and when did it result in oversized drawdowns."
    ),
    "risk_conservative": (
        "Conservative risk management for {symbol}: capital preservation, "
        "downside protection, correlation risks, liquidity constraints. "
        "Past conservative postures — when did caution prevent losses, "
        "and when did it cause missed opportunities."
    ),
    "risk_neutral": (
        "Balanced risk assessment for {symbol}: risk-reward calibration, "
        "position sizing discipline, volatility-adjusted exposure. "
        "Past balanced approaches — what risk-reward profiles proved well-calibrated "
        "and which required adjustment."
    ),
    "research_manager": (
        "Investment research synthesis for {symbol}: debate resolution, "
        "bull-bear balance, key disagreements and convergences. "
        "Past research debates — which side was vindicated, "
        "and what debate patterns preceded correct vs incorrect calls."
    ),
    "trader": (
        "Trade execution context for {symbol}: entry/exit timing, "
        "position sizing, liquidity conditions, spread considerations. "
        "Past trade executions — what entry timing and sizing decisions "
        "produced the best risk-adjusted outcomes."
    ),
    "portfolio_manager": (
        "Comprehensive investment decision synthesis for {symbol}. "
        "Fundamental valuation, technical patterns, sentiment signals, "
        "macro context, risk-reward calibration, and debate resolution. "
        "Past decisions across all dimensions — what patterns of evidence "
        "led to correct calls and what blind spots caused mistakes. "
        "Current market: return {return_5d:+.1f}%, volatility {volatility:.1f}%, "
        "volume ratio {volume_ratio:.2f}x, price at {price_position:.0%} of range."
    ),
}


# ── Profile registry ────────────────────────────────────────────────────────

PROFILES: dict[str, AgentMemoryProfile] = {
    "fundamentals_analyst": AgentMemoryProfile(
        role="fundamentals_analyst",
        chunk_types=["market_context", "thesis", "reflection"],
        interest_tags=[
            "valuation", "earnings", "revenue_growth", "balance_sheet",
            "fcf", "profit_margin", "debt", "pe_ratio", "dividend",
        ],
        sim_weight=0.60, recency_weight=0.20, outcome_weight=0.20,
        cross_ticker=True, max_items=3,
    ),
    "market_analyst": AgentMemoryProfile(
        role="market_analyst",
        chunk_types=["market_context", "reflection"],
        interest_tags=[
            "technical", "price_pattern", "indicator_signal",
            "volume", "volatility", "trend", "support_resistance",
            "macd", "rsi", "moving_average",
        ],
        sim_weight=0.65, recency_weight=0.15, outcome_weight=0.20,
        cross_ticker=True, max_items=4,
    ),
    "sentiment_analyst": AgentMemoryProfile(
        role="sentiment_analyst",
        chunk_types=["thesis", "reflection"],
        interest_tags=[
            "sentiment", "news_sentiment", "social_media", "fear_greed",
            "macro_sentiment", "narrative",
        ],
        sim_weight=0.50, recency_weight=0.35, outcome_weight=0.15,
        cross_ticker=True, max_items=3,
    ),
    "news_analyst": AgentMemoryProfile(
        role="news_analyst",
        chunk_types=["thesis", "reflection"],
        interest_tags=[
            "macro_event", "policy_change", "sector_news", "geopolitical",
            "economic_data", "fed", "earnings_season",
        ],
        sim_weight=0.50, recency_weight=0.35, outcome_weight=0.15,
        cross_ticker=True, max_items=3,
    ),
    "bull_researcher": AgentMemoryProfile(
        role="bull_researcher",
        chunk_types=["thesis", "reflection"],
        interest_tags=[
            "bull_thesis", "upside", "growth_catalyst", "undervaluation",
            "competitive_advantage", "market_share", "innovation",
        ],
        sim_weight=0.30, recency_weight=0.20, outcome_weight=0.50,
        cross_ticker=True, max_items=3,
    ),
    "bear_researcher": AgentMemoryProfile(
        role="bear_researcher",
        chunk_types=["thesis", "reflection"],
        interest_tags=[
            "bear_thesis", "downside", "risk_factor", "overvaluation",
            "headwind", "competition", "regulation",
        ],
        sim_weight=0.30, recency_weight=0.20, outcome_weight=0.50,
        cross_ticker=True, max_items=3,
    ),
    "risk_aggressive": AgentMemoryProfile(
        role="risk_aggressive",
        chunk_types=["thesis", "reflection"],
        interest_tags=[
            "reward_ratio", "upside_asymmetry", "growth_momentum",
            "high_conviction", "catalyst",
        ],
        sim_weight=0.40, recency_weight=0.20, outcome_weight=0.40,
        cross_ticker=False, max_items=3,
    ),
    "risk_conservative": AgentMemoryProfile(
        role="risk_conservative",
        chunk_types=["reflection", "portfolio_context"],
        interest_tags=[
            "drawdown", "tail_risk", "correlation", "liquidity",
            "capital_preservation", "hedge",
        ],
        sim_weight=0.40, recency_weight=0.20, outcome_weight=0.40,
        cross_ticker=False, max_items=3,
    ),
    "risk_neutral": AgentMemoryProfile(
        role="risk_neutral",
        chunk_types=["thesis", "reflection"],
        interest_tags=[
            "risk_reward", "position_sizing", "volatility_adjusted",
            "sharpe", "calibration",
        ],
        sim_weight=0.45, recency_weight=0.25, outcome_weight=0.30,
        cross_ticker=True, max_items=3,
    ),
    "research_manager": AgentMemoryProfile(
        role="research_manager",
        chunk_types=["thesis", "reflection"],
        interest_tags=[
            "debate", "bull_bear", "thesis", "disagreement",
            "research", "synthesis",
        ],
        sim_weight=0.40, recency_weight=0.25, outcome_weight=0.35,
        cross_ticker=True, max_items=4,
    ),
    "trader": AgentMemoryProfile(
        role="trader",
        chunk_types=["portfolio_context", "reflection"],
        interest_tags=[
            "execution", "entry", "sizing", "liquidity",
            "timing", "slippage",
        ],
        sim_weight=0.45, recency_weight=0.30, outcome_weight=0.25,
        cross_ticker=True, max_items=3,
    ),
    "portfolio_manager": AgentMemoryProfile(
        role="portfolio_manager",
        chunk_types=["thesis", "market_context", "reflection", "portfolio_context"],
        interest_tags=["*"],  # PM sees everything
        sim_weight=0.40, recency_weight=0.30, outcome_weight=0.30,
        cross_ticker=True, max_items=5,
    ),
}


def get_profile(role: str) -> AgentMemoryProfile:
    """Return the profile for *role*, falling back to portfolio_manager."""
    return PROFILES.get(role, PROFILES["portfolio_manager"])
