"""Semantic chunking of DecisionRecord into retrievable memory fragments.

Each decision is split into typed chunks so that different agent roles
can retrieve the specific slice they care about: a market analyst hits
the ``market_context`` chunk, a bull researcher hits the ``thesis`` chunk,
and the reflection (added later) enriches every perspective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradingagents.extensions.contracts import DecisionRecord, DecisionOutcome


# Maximum character length for each chunk type.  Kept short so the
# embedding captures the semantic core rather than noise.
_CHUNK_LIMITS: dict[str, int] = {
    "thesis": 800,
    "market_context": 500,
    "portfolio_context": 400,
    "reflection": 600,
}


def _truncate(text: str, limit: int) -> str:
    """Truncate *text* to *limit* characters, breaking at a word boundary."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[:cut if cut > 0 else limit] + "..."


def _format_market_context(record: DecisionRecord) -> str:
    """Render the decision-time market snapshot as a compact structured description."""
    snapshot = record.market_at_decision
    bars = snapshot.bars

    if not bars:
        return f"Symbol: {snapshot.symbol}. No OHLCV data at decision time."

    last = bars[-1]
    parts = [
        f"Symbol: {snapshot.symbol}",
        f"Price: {last.close:.2f} (O:{last.open:.2f} H:{last.high:.2f} L:{last.low:.2f})",
        f"Volume: {last.volume:,.0f}",
        f"Bars available: {len(bars)}",
    ]

    if len(bars) >= 5:
        closes = [b.close for b in bars]
        ret_5 = (closes[-1] / closes[-5] - 1) * 100
        parts.append(f"5-bar return: {ret_5:+.2f}%")

    if len(bars) >= 10:
        avg_vol = sum(b.volume for b in bars[-10:]) / 10
        vol_ratio = last.volume / avg_vol if avg_vol > 0 else 1.0
        parts.append(f"Volume vs 10-bar avg: {vol_ratio:.2f}x")

        highs = [b.high for b in bars[-10:]]
        lows = [b.low for b in bars[-10:]]
        range_pct = (max(highs) / min(lows) - 1) * 100
        parts.append(f"10-bar range: {range_pct:.1f}%")

    return "; ".join(parts)


def _format_portfolio_context(record: DecisionRecord) -> str:
    """Render pre-decision portfolio state as a compact description."""
    pf = record.portfolio_before
    parts = [
        f"Cash: ${pf.cash:,.0f}",
        f"Total equity: ${pf.total_equity:,.0f}",
        f"Positions held: {len(pf.positions)}",
    ]
    for sym, pos in list(pf.positions.items())[:5]:
        parts.append(
            f"  {sym}: {pos.quantity} shares @ ${pos.average_cost:.2f} "
            f"(mkt ${pos.market_value:,.0f}, weight {pos.weight:.0%})"
        )
    return "\n".join(parts)


def _classify_tags(record: DecisionRecord) -> list[str]:
    """Derive agent-relevant tags from the decision rationale text.

    Uses lightweight keyword matching so tags are deterministic and fast.
    New keywords can be added here without changing any other module.
    """
    text = (record.intent.rationale + " " + " ".join(record.intent.warnings)).lower()
    tags: list[str] = []

    _KEYWORD_MAP: list[tuple[list[str], str]] = [
        (["pe ", "p/e ", "price-to-earn", "valuation", "overvalued", "undervalued", "intrinsic"], "valuation"),
        (["revenue", "earnings growth", "eps ", "top line", "bottom line"], "earnings"),
        (["margin", "profitability", "gross margin", "operating margin", "net margin"], "profit_margin"),
        (["debt", "leverage", "balance sheet", "liabilities"], "balance_sheet"),
        (["free cash flow", "fcf", "cash flow", "buyback"], "fcf"),
        (["rsi", "macd", "moving average", "bollinger", "fibonacci"], "technical"),
        (["support", "resistance", "breakout", "breakdown"], "support_resistance"),
        (["volume", "accumulation", "distribution", "volume spike"], "volume"),
        (["volatility", "vix", "beta ", "standard deviation"], "volatility"),
        (["trend", "momentum", "uptrend", "downtrend", "sideways"], "trend"),
        (["sentiment", "mood", "fear", "greed", "social media", "reddit", "twitter"], "sentiment"),
        (["news", "headline", "report", "announcement"], "news_sentiment"),
        (["fed ", "federal reserve", "interest rate", "inflation", "cpi ", "gdp "], "macro_event"),
        (["regulation", "policy", "law", "compliance", "sec "], "regulation"),
        (["bull ", "bullish", "upside", "catalyst", "growth story"], "bull_thesis"),
        (["bear ", "bearish", "downside", "risk", "headwind", "threat"], "bear_thesis"),
        (["drawdown", "tail risk", "black swan", "worst case"], "tail_risk"),
        (["position size", "allocation", "weight", "exposure", "concentration"], "position_sizing"),
        (["entry", "execution", "timing", "slippage", "liquidity"], "execution"),
        (["competitive", "moat", "market share", "disruption"], "competitive_advantage"),
    ]

    for keywords, tag in _KEYWORD_MAP:
        if any(kw in text for kw in keywords):
            tags.append(tag)

    return tags or ["general"]


class DecisionChunker:
    """Split a DecisionRecord into semantically typed chunks for embedding."""

    @staticmethod
    def split(record: DecisionRecord) -> list[dict]:
        """Return a list of ``{"type": str, "content": str}`` dicts."""
        chunks: list[dict] = []

        # Thesis chunk — the core investment rationale
        thesis = record.intent.rationale
        if thesis:
            chunks.append({
                "type": "thesis",
                "content": _truncate(thesis, _CHUNK_LIMITS["thesis"]),
            })

        # Market context chunk — the "what did the world look like" snapshot
        market = _format_market_context(record)
        if market:
            chunks.append({
                "type": "market_context",
                "content": market,
            })

        # Portfolio context chunk — pre-decision positioning
        portfolio = _format_portfolio_context(record)
        if portfolio:
            chunks.append({
                "type": "portfolio_context",
                "content": portfolio,
            })

        return chunks

    @staticmethod
    def build_reflection_chunk(
        record: DecisionRecord, outcome: DecisionOutcome, reflection_text: str
    ) -> dict | None:
        """Create a reflection chunk from an outcome and LLM-generated reflection."""
        if not reflection_text.strip():
            return None

        enriched = (
            f"Outcome: raw return {outcome.holding_period_return:+.2%}"
            if outcome.holding_period_return is not None
            else "Outcome: not yet measured"
        )
        if outcome.max_adverse_move is not None:
            enriched += f", max adverse move {outcome.max_adverse_move:+.2%}"
        if outcome.portfolio_impact is not None:
            enriched += f", portfolio impact {outcome.portfolio_impact:+.2%}"

        return {
            "type": "reflection",
            "content": _truncate(
                f"{enriched}\n\n{reflection_text}",
                _CHUNK_LIMITS["reflection"],
            ),
        }
