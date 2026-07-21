"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "**Rating**: X" / "rating - X"
_RATING_LABEL_RE = re.compile(
    r"(?:\*\*)?rating(?:\*\*)?\s*[:：\-]\s*(?:\*\*)?\s*([A-Za-z]+)",
    re.IGNORECASE,
)

# Chinese / bilingual decision lines (Portfolio Manager often writes these
# when structured-output falls back to free text).
_CN_DECISION_RE = re.compile(
    r"(?:最终)?(?:交易)?决策\s*[:：]\s*(?:\*\*)?\s*(买入|加仓|增持|超配|持有|维持|减持|低配|卖出|清仓)",
    re.IGNORECASE,
)
_CN_RATING_RE = re.compile(
    r"(?:评级|结论)\s*[:：]\s*(?:\*\*)?\s*(买入|加仓|增持|超配|持有|维持|减持|低配|卖出|清仓)",
    re.IGNORECASE,
)

_CN_MAP = {
    "买入": "Buy",
    "加仓": "Overweight",
    "增持": "Overweight",
    "超配": "Overweight",
    "持有": "Hold",
    "维持": "Hold",
    "减持": "Underweight",
    "低配": "Underweight",
    "卖出": "Sell",
    "清仓": "Sell",
}

# Parenthetical English: 卖出 (Sell), 决策：卖出 (Sell)
_PAREN_RATING_RE = re.compile(
    r"[\(（]\s*(Buy|Overweight|Hold|Underweight|Sell)\s*[\)）]",
    re.IGNORECASE,
)

_ACTION_RE = re.compile(
    r"(?:\*\*)?Action(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(Buy|Overweight|Hold|Underweight|Sell)",
    re.IGNORECASE,
)
_RECOMMENDATION_RE = re.compile(
    r"(?:\*\*)?Recommendation(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*"
    r"(Buy|Overweight|Hold|Underweight|Sell)",
    re.IGNORECASE,
)

_FINAL_PROPOSAL_RE = re.compile(
    r"FINAL TRANSACTION PROPOSAL\s*[:：]\s*\**\s*(BUY|SELL|HOLD|OVERWEIGHT|UNDERWEIGHT)\b",
    re.IGNORECASE,
)


def _canon(word: str) -> str | None:
    w = word.strip().strip("*").strip()
    if w.lower() in _RATING_SET:
        # Preserve Overweight / Underweight title-casing
        for r in RATINGS_5_TIER:
            if r.lower() == w.lower():
                return r
    return None


class RatingParseResult(NamedTuple):
    parsed: str | None
    source: str
    error: str | None


def parse_rating_strict(text: str, *, expected_label: str | None = None) -> RatingParseResult:
    """Parse only explicit decision labels and never invent a default Hold."""
    if not text or not str(text).strip():
        return RatingParseResult(None, "none", "empty decision text")
    value = str(text)
    labelled = [
        ("rating_label", _RATING_LABEL_RE),
        ("recommendation_label", _RECOMMENDATION_RE),
        ("action_label", _ACTION_RE),
        ("final_proposal", _FINAL_PROPOSAL_RE),
    ]
    if expected_label:
        key = expected_label.lower()
        labelled.sort(key=lambda item: 0 if key in item[0] else 1)
    for source, pattern in labelled:
        match = pattern.search(value)
        if match:
            parsed = _canon(match.group(1))
            if parsed:
                return RatingParseResult(parsed, source, None)
    for source, pattern in (
        ("chinese_decision", _CN_DECISION_RE),
        ("chinese_rating", _CN_RATING_RE),
    ):
        match = pattern.search(value)
        if match:
            return RatingParseResult(_CN_MAP[match.group(1)], source, None)
    return RatingParseResult(None, "none", "no explicit rating/recommendation/action label")


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Priority (high → low):
    1. Explicit ``Rating:`` / ``**Rating**:`` label
    2. ``FINAL TRANSACTION PROPOSAL: **SELL**``
    3. ``Action: Sell``
    4. Chinese 决策/评级 lines (卖出/减持/…)
    5. Parenthetical ``(Sell)``
    6. First bare English rating word (scan from the end — PM decision usually last)

    Returns a canonical rating string, or ``default`` if nothing matches.
    """
    if not text or not str(text).strip():
        return default
    text = str(text)

    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m:
            got = _canon(m.group(1))
            if got:
                return got

    m = _FINAL_PROPOSAL_RE.search(text)
    if m:
        got = _canon(m.group(1))
        if got:
            return got

    for line in text.splitlines():
        m = _ACTION_RE.search(line)
        if m:
            got = _canon(m.group(1))
            if got:
                return got

    for line in text.splitlines():
        m = _CN_DECISION_RE.search(line) or _CN_RATING_RE.search(line)
        if m:
            return _CN_MAP[m.group(1)]

    # Prefer the last parenthetical rating (often on the decision line).
    parens = list(_PAREN_RATING_RE.finditer(text))
    if parens:
        got = _canon(parens[-1].group(1))
        if got:
            return got

    # Bare English words: scan from bottom so PM "Sell" beats earlier "Buy" mentions.
    for line in reversed(text.splitlines()):
        for word in line.replace("(", " ").replace(")", " ").split():
            got = _canon(word.strip("*:.,，。"))
            if got:
                return got

    # Whole-text Chinese keyword fallback (last occurrence wins).
    last_cn = None
    for cn, eng in _CN_MAP.items():
        for m in re.finditer(re.escape(cn), text):
            last_cn = (m.start(), eng)
    if last_cn is not None:
        # If both 买入 and 卖出 appear, prefer the later one.
        best = None
        for cn, eng in _CN_MAP.items():
            for m in re.finditer(re.escape(cn), text):
                if best is None or m.start() >= best[0]:
                    best = (m.start(), eng)
        if best:
            return best[1]

    return default
