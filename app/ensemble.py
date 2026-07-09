"""Signal fusion: every detector emits Signals; we fuse them in logit space.

Each signal is an independent estimate of P(AI-generated) with a weight
reflecting how much we trust that family of evidence. Fusing in logit space
(weighted mean of log-odds) keeps one confident-but-wrong signal from
saturating the result, while letting decisive evidence (e.g. a C2PA
"trainedAlgorithmicMedia" manifest) carry a large weight and dominate.

The weights are hand-calibrated priors, deliberately stored on every persisted
analysis — the feedback table gives ground truth, so they can be re-fit (or a
proper model trained) once enough labelled data accumulates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Signal:
    name: str
    label: str                      # human-readable, shown in the UI
    score: float | None             # P(AI) in [0,1]; None = signal unavailable
    weight: float                   # trust in this signal family
    detail: str = ""                # one-line explanation shown in the UI
    raw: dict = field(default_factory=dict)  # underlying measurements (persisted)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "score": None if self.score is None else round(self.score, 4),
            "weight": self.weight,
            "detail": self.detail,
            "raw": self.raw,
        }


def _logit(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def sig_score(value: float, center: float, scale: float, invert: bool = False) -> float:
    """Map a raw measurement onto [0,1] with a soft threshold.

    Score rises as `value` rises past `center` (or falls, when invert=True);
    `scale` controls how quickly the sigmoid commits.
    """
    x = (value - center) / scale
    if invert:
        x = -x
    return _sigmoid(x)


def combine(signals: list[Signal]) -> dict:
    """Weighted logit-mean of available signals -> percent + agreement stats."""
    live = [s for s in signals if s.score is not None and s.weight > 0]
    if not live:
        return {"percent": None, "confidence": "none", "signals": [s.as_dict() for s in signals]}

    total_w = sum(s.weight for s in live)
    fused_logit = sum(s.weight * _logit(s.score) for s in live) / total_w
    percent = _sigmoid(fused_logit) * 100

    # Confidence: how much evidence we had, how decisive it was, and whether it
    # agreed (weight-aware, so a weak dissenting signal can't mark a decisive
    # verdict as "mixed").
    wstd = math.sqrt(
        sum(s.weight * (_logit(s.score) - fused_logit) ** 2 for s in live) / total_w
    )
    if total_w < 1.0 or len(live) < 2:
        confidence = "low"
    elif abs(fused_logit) >= 1.4:
        confidence = "high"
    elif abs(fused_logit) >= 0.6:
        confidence = "medium"
    elif wstd > 2.2:
        confidence = "mixed"          # strong signals genuinely disagree
    else:
        confidence = "low"

    return {
        "percent": round(percent, 1),
        "confidence": confidence,
        "fused_logit": round(fused_logit, 3),
        "signals_used": len(live),
        "signals": [s.as_dict() for s in signals],
    }
