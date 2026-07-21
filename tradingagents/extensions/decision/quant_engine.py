"""Deterministic multi-horizon features and q_score from OHLCV bars."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tradingagents.extensions.contracts import MarketBar, MarketSnapshot


@dataclass(frozen=True)
class QuantSignal:
    """Purely deterministic view of the market for fusion / diagnostics."""

    q_score: float
    suggested_weight: float
    confidence: float
    features: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "q_score": self.q_score,
            "suggested_weight": self.suggested_weight,
            "confidence": self.confidence,
            "features": dict(self.features),
            "warnings": list(self.warnings),
        }


def _closes(bars: list[MarketBar]) -> np.ndarray:
    return np.asarray([b.close for b in bars], dtype=float)


def _opens(bars: list[MarketBar]) -> np.ndarray:
    return np.asarray([b.open for b in bars], dtype=float)


def _volumes(bars: list[MarketBar]) -> np.ndarray:
    return np.asarray([b.volume for b in bars], dtype=float)


def _safe_ret(closes: np.ndarray, window: int) -> float | None:
    if len(closes) <= window or closes[-1 - window] <= 0:
        return None
    return float(closes[-1] / closes[-1 - window] - 1.0)


def _sma_ratio(closes: np.ndarray, window: int) -> float | None:
    if len(closes) < window:
        return None
    sma = float(np.mean(closes[-window:]))
    if sma <= 0:
        return None
    return float(closes[-1] / sma - 1.0)


def _realized_vol(closes: np.ndarray, window: int) -> float | None:
    if len(closes) < window + 1:
        return None
    rets = np.diff(np.log(closes[-(window + 1) :]))
    if len(rets) == 0:
        return None
    return float(np.std(rets, ddof=1)) if len(rets) > 1 else float(abs(rets[0]))


def _volume_z(volumes: np.ndarray, window: int = 20) -> float | None:
    if len(volumes) < window:
        return None
    hist = volumes[-window:]
    mu = float(np.mean(hist))
    sigma = float(np.std(hist, ddof=1)) if window > 1 else 0.0
    if sigma < 1e-12:
        return 0.0
    return float((volumes[-1] - mu) / sigma)


def _gap(opens: np.ndarray, closes: np.ndarray) -> float | None:
    if len(closes) < 2 or closes[-2] <= 0:
        return None
    return float(opens[-1] / closes[-2] - 1.0)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def compute_quant_signal(
    market: MarketSnapshot,
    *,
    max_weight: float = 0.35,
    current_weight: float = 0.0,
) -> QuantSignal:
    """Build a QuantSignal from ``market.bars`` (must already be as_of-safe)."""
    bars = market.bars
    warnings: list[str] = []
    features: dict[str, float] = {}

    if len(bars) < 5:
        warnings.append("insufficient_bars")
        return QuantSignal(
            q_score=0.0,
            suggested_weight=current_weight,
            confidence=0.0,
            features=features,
            warnings=warnings,
        )

    closes = _closes(bars)
    opens = _opens(bars)
    volumes = _volumes(bars)

    for name, window in (("ret_5", 5), ("ret_20", 20), ("ret_60", 60)):
        val = _safe_ret(closes, window)
        if val is None:
            warnings.append(f"missing_{name}")
        else:
            features[name] = val

    for name, window in (("sma_ratio_20", 20), ("sma_ratio_60", 60)):
        val = _sma_ratio(closes, window)
        if val is None:
            warnings.append(f"missing_{name}")
        else:
            features[name] = val

    vol20 = _realized_vol(closes, 20)
    if vol20 is None:
        warnings.append("missing_vol_20")
    else:
        features["vol_20"] = vol20

    vz = _volume_z(volumes, 20)
    if vz is None:
        warnings.append("missing_volume_z_20")
    else:
        features["volume_z_20"] = vz

    gap = _gap(opens, closes)
    if gap is None:
        warnings.append("missing_gap_1")
    else:
        features["gap_1"] = gap

    # Map features into a bounded score. Weights favour medium-horizon trend.
    components: list[tuple[float, float]] = []
    if "ret_5" in features:
        components.append((_clip(features["ret_5"] / 0.05), 0.15))
    if "ret_20" in features:
        components.append((_clip(features["ret_20"] / 0.10), 0.30))
    if "ret_60" in features:
        components.append((_clip(features["ret_60"] / 0.20), 0.15))
    if "sma_ratio_20" in features:
        components.append((_clip(features["sma_ratio_20"] / 0.05), 0.25))
    if "sma_ratio_60" in features:
        components.append((_clip(features["sma_ratio_60"] / 0.08), 0.10))
    if "volume_z_20" in features:
        # Mild confirmation only — extreme volume alone is not directional.
        components.append((_clip(features["volume_z_20"] / 3.0) * 0.35, 0.05))

    if not components:
        warnings.append("no_score_components")
        return QuantSignal(
            q_score=0.0,
            suggested_weight=current_weight,
            confidence=0.0,
            features=features,
            warnings=warnings,
        )

    weight_sum = sum(w for _, w in components)
    q_score = sum(v * w for v, w in components) / weight_sum
    q_score = _clip(q_score)

    # High realized vol reduces confidence and pulls suggested weight to cash.
    vol_penalty = 0.0
    if vol20 is not None:
        # ~1% daily vol is "normal"; above that shrink confidence.
        vol_penalty = _clip((vol20 - 0.01) / 0.02, 0.0, 1.0)
        features["vol_penalty"] = vol_penalty

    coverage = min(1.0, len(bars) / 60.0)
    component_coverage = len(components) / 6.0
    confidence = float(
        max(0.0, min(1.0, 0.55 * coverage + 0.45 * component_coverage - 0.35 * vol_penalty))
    )

    raw_weight = (q_score + 1.0) / 2.0 * max_weight
    # Blend toward current when confidence is low (stability).
    suggested = confidence * raw_weight + (1.0 - confidence) * current_weight
    suggested = float(max(0.0, min(max_weight, suggested)))

    return QuantSignal(
        q_score=float(q_score),
        suggested_weight=suggested,
        confidence=confidence,
        features=features,
        warnings=warnings,
    )


def multi_horizon_ohlcv_summary(market: MarketSnapshot) -> dict[str, Any]:
    """Tool-friendly numerical summary across short/medium/long windows."""
    signal = compute_quant_signal(market)
    closes = _closes(market.bars) if market.bars else np.asarray([])
    summary = {
        "symbol": market.symbol,
        "as_of": market.as_of.isoformat(),
        "n_bars": len(market.bars),
        "last_close": float(closes[-1]) if len(closes) else None,
        "q_score": signal.q_score,
        "quant_confidence": signal.confidence,
        "features": signal.features,
        "warnings": signal.warnings,
        "horizons": {},
    }
    for label, window in (("short", 5), ("medium", 20), ("long", 60)):
        summary["horizons"][label] = {
            "return": _safe_ret(closes, window) if len(closes) else None,
            "sma_ratio": _sma_ratio(closes, window) if len(closes) else None,
            "realized_vol": _realized_vol(closes, window) if len(closes) else None,
        }
    if market.bars:
        summary["volume_z_20"] = _volume_z(_volumes(market.bars), 20)
        summary["gap_1"] = _gap(_opens(market.bars), closes)
    return summary
