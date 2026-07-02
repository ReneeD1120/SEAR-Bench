from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .evidence import FactorEvidence


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def evidence_to_features(evidence: FactorEvidence) -> np.ndarray:
    def _clean(x: float) -> float:
        return 0.0 if x is None or math.isnan(x) else float(x)

    return np.array(
        [
            abs(_clean(evidence.ic)),
            np.tanh(abs(_clean(evidence.ic_ir)) / 5.0),
            _clean(evidence.win_rate) - 0.5,
            _clean(evidence.stability),
            np.tanh(_clean(evidence.regime_contrast) * 5.0),
            np.tanh(abs(_clean(evidence.regime_ic_high_vol)) * 20.0),
            np.tanh(abs(_clean(evidence.regime_ic_low_vol)) * 20.0),
            np.tanh(_clean(evidence.strategy_sharpe) / 3.0),
            np.tanh(_clean(evidence.strategy_cum_return)),
            min(0.0, _clean(evidence.strategy_max_drawdown)),
        ],
        dtype=float,
    )


@dataclass
class LinearEvidenceJudge:
    weights: np.ndarray
    bias: float
    threshold: float = 0.5

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.3,
        steps: int = 400,
        l2: float = 0.01,
    ) -> "LinearEvidenceJudge":
        n, d = X.shape
        weights = np.zeros(d, dtype=float)
        bias = 0.0
        y = y.astype(float)
        for _ in range(steps):
            logits = X @ weights + bias
            probs = _sigmoid(logits)
            error = probs - y
            grad_w = (X.T @ error) / max(1, n) + l2 * weights
            grad_b = float(error.mean())
            weights -= lr * grad_w
            bias -= lr * grad_b
        return cls(weights=weights, bias=bias)

    def score(self, evidence: FactorEvidence) -> float:
        x = evidence_to_features(evidence)
        return float(_sigmoid(np.array([x @ self.weights + self.bias]))[0])

    def predict(self, evidence: FactorEvidence) -> dict[str, object]:
        score = self.score(evidence)
        active_regime = "high_vol" if abs(evidence.regime_ic_high_vol) > abs(evidence.regime_ic_low_vol) else "low_vol"
        return {
            "decision": "keep" if score >= self.threshold else "drop",
            "active_regime": active_regime,
            "confidence": float(score),
            "rationale": "learned evidence scorer over IC, IR, stability and regime contrast",
        }


def rule_based_judge(evidence: FactorEvidence, *, min_ic: float = 0.02, min_win_rate: float = 0.52) -> dict[str, object]:
    strategy_score = max(0.0, min(1.0, evidence.strategy_sharpe / 3.0)) if not math.isnan(evidence.strategy_sharpe) else 0.0
    drawdown_penalty = min(0.2, abs(evidence.strategy_max_drawdown)) if not math.isnan(evidence.strategy_max_drawdown) else 0.0
    score = (
        0.35 * abs(evidence.ic)
        + 0.20 * min(1.0, abs(evidence.ic_ir) / 5.0 if not math.isnan(evidence.ic_ir) else 0.0)
        + 0.15 * evidence.stability
        + 0.10 * min(1.0, evidence.regime_contrast * 5.0 if not math.isnan(evidence.regime_contrast) else 0.0)
        + 0.25 * strategy_score
        - 0.10 * drawdown_penalty
    )
    active_regime = "high_vol" if abs(evidence.regime_ic_high_vol) > abs(evidence.regime_ic_low_vol) else "low_vol"
    keep = (
        not math.isnan(evidence.ic)
        and abs(evidence.ic) >= min_ic
        and not math.isnan(evidence.win_rate)
        and evidence.win_rate >= min_win_rate
        and score >= 0.25
    )
    return {
        "decision": "keep" if keep else "drop",
        "active_regime": active_regime,
        "confidence": float(min(1.0, max(0.0, score))),
        "rationale": "thresholded heuristic over IC, stability, regime contrast and in-sample strategy evidence",
    }


def fit_judge_from_rows(rows: pd.DataFrame, label_col: str = "label_keep") -> LinearEvidenceJudge:
    X = np.vstack(rows["evidence"].map(evidence_to_features).to_list())
    y = rows[label_col].to_numpy(dtype=float)
    return LinearEvidenceJudge.fit(X, y)
