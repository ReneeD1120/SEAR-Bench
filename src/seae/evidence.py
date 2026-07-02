from __future__ import annotations

from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 8:
        return float("nan")
    return float(x.iloc[:, 0].corr(x.iloc[:, 1]))


def _safe_mean(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean()) if len(x) else float("nan")


def _safe_std(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


@dataclass(frozen=True)
class FactorEvidence:
    symbol: str
    factor_name: str
    horizon: int
    ic: float
    ic_ir: float
    win_rate: float
    stability: float
    regime_ic_high_vol: float
    regime_ic_low_vol: float
    regime_contrast: float
    n_obs: int
    train_ic: float = float("nan")
    test_ic: float = float("nan")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def extract_factor_evidence(
    df: pd.DataFrame,
    factor: pd.Series,
    *,
    symbol: str,
    factor_name: str,
    horizon: int = 5,
    vol_window: int = 20,
) -> FactorEvidence:
    feature = factor.astype(float)
    future_ret = df["close"].shift(-horizon) / df["close"] - 1.0

    aligned = pd.concat([feature, future_ret], axis=1).dropna()
    if aligned.empty:
        return FactorEvidence(
            symbol=symbol,
            factor_name=factor_name,
            horizon=horizon,
            ic=float("nan"),
            ic_ir=float("nan"),
            win_rate=float("nan"),
            stability=float("nan"),
            regime_ic_high_vol=float("nan"),
            regime_ic_low_vol=float("nan"),
            regime_contrast=float("nan"),
            n_obs=0,
        )

    ic = _safe_corr(feature, future_ret)
    rolling_ic = aligned.iloc[:, 0].rolling(vol_window).corr(aligned.iloc[:, 1]).dropna()
    ic_ir = float(_safe_mean(rolling_ic) / (rolling_ic.std(ddof=1) + 1e-12)) if len(rolling_ic) else float("nan")
    win_rate = float((aligned.iloc[:, 0] * aligned.iloc[:, 1] > 0).mean())
    stability = float(1.0 / (1.0 + aligned.iloc[:, 0].rolling(vol_window).std().mean()))

    vol = df["close"].pct_change().rolling(vol_window).std()
    hi_mask = vol > vol.median()
    lo_mask = ~hi_mask
    regime_ic_high_vol = _safe_corr(feature[hi_mask], future_ret[hi_mask])
    regime_ic_low_vol = _safe_corr(feature[lo_mask], future_ret[lo_mask])
    regime_contrast = float(abs(regime_ic_high_vol - regime_ic_low_vol)) if not (np.isnan(regime_ic_high_vol) or np.isnan(regime_ic_low_vol)) else float("nan")

    return FactorEvidence(
        symbol=symbol,
        factor_name=factor_name,
        horizon=horizon,
        ic=ic,
        ic_ir=ic_ir,
        win_rate=win_rate,
        stability=stability,
        regime_ic_high_vol=regime_ic_high_vol,
        regime_ic_low_vol=regime_ic_low_vol,
        regime_contrast=regime_contrast,
        n_obs=int(len(aligned)),
    )


def split_time_evidence(
    df: pd.DataFrame,
    factor: pd.Series,
    *,
    symbol: str,
    factor_name: str,
    horizon: int = 5,
    split_ratio: float = 0.7,
) -> FactorEvidence:
    cutoff = int(len(df) * split_ratio)
    train_df = df.iloc[:cutoff].copy()
    test_df = df.iloc[cutoff:].copy()
    train_factor = factor.iloc[:cutoff]
    test_factor = factor.iloc[cutoff:]

    train_ev = extract_factor_evidence(
        train_df,
        train_factor,
        symbol=symbol,
        factor_name=factor_name,
        horizon=horizon,
    )
    test_ev = extract_factor_evidence(
        test_df,
        test_factor,
        symbol=symbol,
        factor_name=factor_name,
        horizon=horizon,
    )

    return FactorEvidence(
        symbol=symbol,
        factor_name=factor_name,
        horizon=horizon,
        ic=train_ev.ic,
        ic_ir=train_ev.ic_ir,
        win_rate=train_ev.win_rate,
        stability=train_ev.stability,
        regime_ic_high_vol=train_ev.regime_ic_high_vol,
        regime_ic_low_vol=train_ev.regime_ic_low_vol,
        regime_contrast=train_ev.regime_contrast,
        n_obs=train_ev.n_obs,
        train_ic=train_ev.ic,
        test_ic=test_ev.ic,
    )

