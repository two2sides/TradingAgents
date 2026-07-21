"""LangChain-ready tool registry for Developer C."""

from .ohlcv_tools import REACT_OHLCV_TOOLS
from .sentiment_tools import REACT_SENTIMENT_TOOLS
from .debate_tools import DEBATE_REACT_TOOLS

REACT_TOOLS = [*REACT_OHLCV_TOOLS, *REACT_SENTIMENT_TOOLS, *DEBATE_REACT_TOOLS]

__all__ = [
    "REACT_OHLCV_TOOLS",
    "REACT_SENTIMENT_TOOLS",
    "DEBATE_REACT_TOOLS",
    "REACT_TOOLS",
]
