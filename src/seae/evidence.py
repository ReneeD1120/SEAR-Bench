from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import warnings

import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8:
        return float("nan")
    left = x.iloc[:, 0].to_numpy(dtype=float)
    right = x.iloc[:, 1].to_numpy(dtype=float)
    left_std = left.std(ddof=1)
    right_std = right.std(ddof=1)
    if left_std <= 1e-12 or right_std <= 1e-12:
        return float("nan")
    covariance = float(np.dot(left - left.mean(), right - right.mean()) / (len(left) - 1))
    return covariance / float(left_std * right_std)


def _safe_mean(x: pd.Series) -> float:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.mean()) if len(x) else float("nan")


def _safe_std(x: pd.Series) -> float:
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
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
    strategy_mean_return: float = float("nan")
    strategy_sharpe: float = float("nan")
    strategy_cum_return: float = float("nan")
    strategy_max_drawdown: float = float("nan")
    test_strategy_mean_return: float = float("nan")
    test_strategy_sharpe: float = float("nan")
    test_strategy_cum_return: float = float("nan")
    test_strategy_max_drawdown: float = float("nan")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def _strategy_stats(returns: pd.Series, *, horizon: int) -> dict[str, float]:
    x = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 30:
        return {
            "mean_return": float("nan"),
            "sharpe": float("nan"),
            "cum_return": float("nan"),
            "max_drawdown": float("nan"),
        }
    clipped = x.clip(-0.1, 0.1)
    scale = np.sqrt(252.0)
    std = clipped.std(ddof=1)
    if std <= 1e-6:
        sharpe = float("nan")
    else:
        sharpe = float(np.clip(clipped.mean() / std * scale, -10.0, 10.0))
    equity = (1.0 + clipped).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "mean_return": float(clipped.mean()),
        "sharpe": sharpe,
        "cum_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }


def _directional_strategy_returns(
    df: pd.DataFrame,
    factor: pd.Series,
    *,
    horizon: int,
    orientation: float,
    signal_window: int = 20,
) -> pd.Series:
    feature = factor.astype(float).replace([np.inf, -np.inf], np.nan)
    rolling_mean = feature.rolling(signal_window, min_periods=max(5, signal_window // 2)).mean()
    rolling_std = feature.rolling(signal_window, min_periods=max(5, signal_window // 2)).std()
    signal = orientation * (feature - rolling_mean) / (rolling_std + 1e-12)
    position = np.sign(signal).replace(0.0, np.nan)
    future_ret = (df["close"].shift(-horizon) / df["close"] - 1.0).replace([np.inf, -np.inf], np.nan)
    return position * future_ret.clip(-0.3, 0.3) / max(1, horizon)


def extract_factor_evidence(
    df: pd.DataFrame,
    factor: pd.Series,
    *,
    symbol: str,
    factor_name: str,
    horizon: int = 5,
    vol_window: int = 20,
    strategy_orientation: float | None = None,
) -> FactorEvidence:
    feature = factor.astype(float).replace([np.inf, -np.inf], np.nan)
    future_ret = (df["close"].shift(-horizon) / df["close"] - 1.0).replace([np.inf, -np.inf], np.nan)

    aligned = pd.concat([feature, future_ret], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rolling_ic = aligned.iloc[:, 0].rolling(vol_window).corr(aligned.iloc[:, 1]).replace([np.inf, -np.inf], np.nan).dropna()
    ic_ir = float(_safe_mean(rolling_ic) / (rolling_ic.std(ddof=1) + 1e-12)) if len(rolling_ic) else float("nan")
    win_rate = float((aligned.iloc[:, 0] * aligned.iloc[:, 1] > 0).mean())
    stability_std = aligned.iloc[:, 0].rolling(vol_window).std().replace([np.inf, -np.inf], np.nan).mean()
    stability = float(1.0 / (1.0 + stability_std))

    vol = df["close"].pct_change().replace([np.inf, -np.inf], np.nan).rolling(vol_window).std()
    hi_mask = vol > vol.median()
    lo_mask = ~hi_mask
    regime_ic_high_vol = _safe_corr(feature[hi_mask], future_ret[hi_mask])
    regime_ic_low_vol = _safe_corr(feature[lo_mask], future_ret[lo_mask])
    regime_contrast = float(abs(regime_ic_high_vol - regime_ic_low_vol)) if not (np.isnan(regime_ic_high_vol) or np.isnan(regime_ic_low_vol)) else float("nan")

    orientation = strategy_orientation
    if orientation is None:
        orientation = 1.0 if np.isnan(ic) or ic >= 0 else -1.0
    strategy_returns = _directional_strategy_returns(df, feature, horizon=horizon, orientation=float(orientation))
    strategy = _strategy_stats(strategy_returns, horizon=horizon)

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
        strategy_mean_return=strategy["mean_return"],
        strategy_sharpe=strategy["sharpe"],
        strategy_cum_return=strategy["cum_return"],
        strategy_max_drawdown=strategy["max_drawdown"],
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
    orientation = 1.0 if np.isnan(train_ev.ic) or train_ev.ic >= 0 else -1.0
    test_ev = extract_factor_evidence(
        test_df,
        test_factor,
        symbol=symbol,
        factor_name=factor_name,
        horizon=horizon,
        strategy_orientation=orientation,
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
        strategy_mean_return=train_ev.strategy_mean_return,
        strategy_sharpe=train_ev.strategy_sharpe,
        strategy_cum_return=train_ev.strategy_cum_return,
        strategy_max_drawdown=train_ev.strategy_max_drawdown,
        test_strategy_mean_return=test_ev.strategy_mean_return,
        test_strategy_sharpe=test_ev.strategy_sharpe,
        test_strategy_cum_return=test_ev.strategy_cum_return,
        test_strategy_max_drawdown=test_ev.strategy_max_drawdown,
    )
