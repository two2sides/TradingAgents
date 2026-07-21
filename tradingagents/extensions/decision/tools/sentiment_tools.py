"""Sentiment scoring tools (lexicon) exposable to ReAct agents."""

from __future__ import annotations

import json
from typing import Annotated

from tradingagents.extensions.decision.sentiment_lexicon import score_documents, score_text

try:
    from langchain_core.tools import tool as langchain_tool
except ImportError:  # pragma: no cover
    langchain_tool = None


def score_news_sentiment(text: str) -> dict:
    """Score a news headline/body with the finance lexicon."""
    return score_text(text).to_dict()


def score_social_sentiment(posts_json: str) -> dict:
    """Score a JSON list of social posts / comments."""
    try:
        posts = json.loads(posts_json) if posts_json.strip().startswith("[") else [posts_json]
    except json.JSONDecodeError:
        posts = [posts_json]
    if isinstance(posts, dict):
        posts = [str(posts)]
    docs = [str(p) for p in posts]
    return score_documents(docs).to_dict()


def _tool_news(
    text: Annotated[str, "news headline or article text to score"],
) -> str:
    """Score news text with a finance lexicon into a bounded sentiment value."""
    return json.dumps(score_news_sentiment(text), ensure_ascii=False)


def _tool_social(
    posts_json: Annotated[str, "JSON array of social posts, or a single string"],
) -> str:
    """Score social posts with a finance lexicon into a bounded sentiment value."""
    return json.dumps(score_social_sentiment(posts_json), ensure_ascii=False)


if langchain_tool is not None:
    score_news_sentiment_tool = langchain_tool(_tool_news)
    score_social_sentiment_tool = langchain_tool(_tool_social)
else:  # pragma: no cover
    score_news_sentiment_tool = _tool_news
    score_social_sentiment_tool = _tool_social


REACT_SENTIMENT_TOOLS = [
    score_news_sentiment_tool,
    score_social_sentiment_tool,
]
