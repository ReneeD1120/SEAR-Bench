from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorEvidence:
    symbol: str
    factor_name: str
    horizon: int
    ic: float
    ic_ir: float
    win_rate: float
    stability: float
    n_obs: int
    regime_summary: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 5:
        return float("nan")
    return float(x.iloc[:, 0].corr(x.iloc[:, 1]))


def summarize_factor(df: pd.DataFrame, factor: pd.Series, horizon: int = 5) -> FactorEvidence:
    f = factor.astype(float)
    y = df["close"].shift(-horizon) / df["close"] - 1.0
    ic = _safe_corr(f, y)
    valid = pd.concat([f, y], axis=1).dropna()
    if len(valid) >= 10:
        rolling = valid.iloc[:, 0].rolling(20).corr(valid.iloc[:, 1]).dropna()
        ic_ir = float(rolling.mean() / (rolling.std(ddof=1) + 1e-12)) if len(rolling) else float("nan")
        win_rate = float((valid.iloc[:, 0] * valid.iloc[:, 1] > 0).mean())
        stability = float(1.0 / (1.0 + valid.iloc[:, 0].rolling(20).std().mean()))
    else:
        ic_ir = float("nan")
        win_rate = float("nan")
        stability = float("nan")

    vol = df["close"].pct_change().rolling(20).std()
    regime_hi = vol > vol.median()
    regime_lo = ~regime_hi
    regime_summary = {
        "ic_high_vol": _safe_corr(f[regime_hi], y[regime_hi]),
        "ic_low_vol": _safe_corr(f[regime_lo], y[regime_lo]),
        "obs_high_vol": float(regime_hi.sum()),
        "obs_low_vol": float(regime_lo.sum()),
    }
    return FactorEvidence(
        symbol="",
        factor_name="",
        horizon=horizon,
        ic=ic,
        ic_ir=ic_ir,
        win_rate=win_rate,
        stability=stability,
        n_obs=int(valid.shape[0]),
        regime_summary=regime_summary,
    )


def rule_based_judge(evidence: FactorEvidence, ic_threshold: float = 0.02, win_rate_threshold: float = 0.52) -> dict[str, object]:
    keep = (
        not math.isnan(evidence.ic)
        and abs(evidence.ic) >= ic_threshold
        and not math.isnan(evidence.win_rate)
        and evidence.win_rate >= win_rate_threshold
    )
    active_regime = "high_vol" if (evidence.regime_summary.get("ic_high_vol", 0.0) or 0.0) > (evidence.regime_summary.get("ic_low_vol", 0.0) or 0.0) else "low_vol"
    return {
        "decision": "keep" if keep else "drop",
        "active_regime": active_regime,
        "confidence": float(min(1.0, abs(evidence.ic) * 10 if not math.isnan(evidence.ic) else 0.0)),
        "rationale": "threshold rule over IC and win rate",
    }

