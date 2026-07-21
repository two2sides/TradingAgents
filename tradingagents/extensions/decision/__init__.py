"""Analyst-facing tools package (OHLCV helpers, sentiment lexicon wrappers).

Decision-gate / hybrid fusion was removed; this package only serves ReAct analysts.
"""

from tradingagents.extensions.decision.tools import REACT_TOOLS

__all__ = ["REACT_TOOLS"]
