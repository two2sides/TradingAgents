"""Lexicon-based sentiment scoring for news / social snippets.

No ML dependency: a small bilingual word list plus simple polarity rules.
Good enough as a stable numeric Tool input for ReAct agents and for
soft-perturbing the quant score inside fusion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Compact finance-oriented lexicon (EN + a few CN tokens).
_POSITIVE = {
    "beat",
    "beats",
    "surge",
    "surges",
    "rally",
    "bull",
    "bullish",
    "upgrade",
    "upgraded",
    "growth",
    "record",
    "strong",
    "outperform",
    "profit",
    "profits",
    "gain",
    "gains",
    "optimistic",
    "突破",
    "上涨",
    "增长",
    "超预期",
    "利好",
    "看好",
}

_NEGATIVE = {
    "miss",
    "misses",
    "plunge",
    "plunges",
    "crash",
    "bear",
    "bearish",
    "downgrade",
    "downgraded",
    "lawsuit",
    "fraud",
    "weak",
    "loss",
    "losses",
    "cut",
    "cuts",
    "layoff",
    "layoffs",
    "recession",
    "fear",
    "下跌",
    "暴跌",
    "亏损",
    "低于预期",
    "利空",
    "看空",
}

_NEGATORS = {"not", "no", "never", "n't", "without", "不", "未", "没有"}

_TOKEN_RE = re.compile(r"[A-Za-z']+|[\u4e00-\u9fff]{1,4}")


@dataclass(frozen=True)
class SentimentScore:
    score: float  # [-1, 1]
    positive_hits: int
    negative_hits: int
    token_count: int
    label: str

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "positive_hits": self.positive_hits,
            "negative_hits": self.negative_hits,
            "token_count": self.token_count,
            "label": self.label,
        }


def _label(score: float) -> str:
    if score >= 0.25:
        return "bullish"
    if score <= -0.25:
        return "bearish"
    return "neutral"


def score_text(text: str) -> SentimentScore:
    """Score a free-text blob into a bounded sentiment value."""
    if not text or not text.strip():
        return SentimentScore(0.0, 0, 0, 0, "neutral")

    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    pos = 0
    neg = 0
    for i, tok in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in _NEGATORS
        if tok in _POSITIVE:
            if negated:
                neg += 1
            else:
                pos += 1
        elif tok in _NEGATIVE:
            if negated:
                pos += 1
            else:
                neg += 1

    total = pos + neg
    if total == 0:
        return SentimentScore(0.0, 0, 0, len(tokens), "neutral")

    raw = (pos - neg) / total
    # Mild length dampening so one-word headlines don't dominate forever.
    damp = 1.0 - math.exp(-total / 3.0)
    score = max(-1.0, min(1.0, raw * damp))
    return SentimentScore(score, pos, neg, len(tokens), _label(score))


def score_documents(documents: list[str]) -> SentimentScore:
    """Average document-level scores (empty docs ignored)."""
    scored = [score_text(doc) for doc in documents if doc and doc.strip()]
    if not scored:
        return SentimentScore(0.0, 0, 0, 0, "neutral")
    avg = sum(s.score for s in scored) / len(scored)
    pos = sum(s.positive_hits for s in scored)
    neg = sum(s.negative_hits for s in scored)
    toks = sum(s.token_count for s in scored)
    return SentimentScore(max(-1.0, min(1.0, avg)), pos, neg, toks, _label(avg))
