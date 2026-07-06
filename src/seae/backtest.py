from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from .data import load_zip_archive
from .factors import add_basic_features, factor_series_map


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    zip_path: str | Path
    decisions_path: str | Path
    limit: int | None = 10
    decision_col: str = "llm_decision"
    keep_value: str = "keep"
    split_ratio: float = 0.7
    signal_window: int = 20
    long_quantile: float = 0.3
    cost_bps: float = 5.0
    factor_weighting: str = "both"
    output_dir: str | Path | None = "outputs/portfolio_backtest"


def _safe_sharpe(returns: pd.Series) -> float:
    x = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return float("nan")
    std = x.std(ddof=1)
    if std <= 1e-12:
        return float("nan")
    return float(x.mean() / std * np.sqrt(252.0))


def _max_drawdown(returns: pd.Series) -> float:
    x = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if x.empty:
        return float("nan")
    equity = (1.0 + x).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def _summary_from_daily(daily: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    if daily.empty:
        return {
            f"{prefix}_n_days": 0.0,
            f"{prefix}_gross_cum_return": float("nan"),
            f"{prefix}_net_cum_return": float("nan"),
            f"{prefix}_gross_sharpe": float("nan"),
            f"{prefix}_net_sharpe": float("nan"),
            f"{prefix}_net_max_drawdown": float("nan"),
            f"{prefix}_avg_daily_turnover": float("nan"),
            f"{prefix}_avg_daily_cost": float("nan"),
        }
    gross = daily["gross_return"]
    net = daily["net_return"]
    return {
        f"{prefix}_n_days": float(len(daily)),
        f"{prefix}_gross_mean_daily_return": float(gross.mean()),
        f"{prefix}_net_mean_daily_return": float(net.mean()),
        f"{prefix}_gross_cum_return": float((1.0 + gross.fillna(0.0)).prod() - 1.0),
        f"{prefix}_net_cum_return": float((1.0 + net.fillna(0.0)).prod() - 1.0),
        f"{prefix}_gross_sharpe": _safe_sharpe(gross),
        f"{prefix}_net_sharpe": _safe_sharpe(net),
        f"{prefix}_gross_max_drawdown": _max_drawdown(gross),
        f"{prefix}_net_max_drawdown": _max_drawdown(net),
        f"{prefix}_avg_daily_turnover": float(daily["turnover"].mean()),
        f"{prefix}_avg_daily_cost": float(daily["cost"].mean()),
        f"{prefix}_avg_long_count": float(daily["long_count"].mean()),
        f"{prefix}_avg_short_count": float(daily["short_count"].mean()),
    }


def _load_kept_decisions(config: PortfolioBacktestConfig) -> pd.DataFrame:
    decisions = pd.read_csv(config.decisions_path)
    required = {"symbol", "factor_name", config.decision_col}
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"decisions file is missing required columns: {sorted(missing)}")
    keep_mask = decisions[config.decision_col].astype(str).str.lower() == config.keep_value.lower()
    kept = decisions.loc[keep_mask].copy()
    if kept.empty:
        return kept
    if "train_ic" not in kept.columns:
        kept["train_ic"] = np.nan
    if "llm_confidence" not in kept.columns:
        kept["llm_confidence"] = 1.0
    return kept.drop_duplicates(["symbol", "factor_name"]).reset_index(drop=True)


def _factor_signal(series: pd.Series, *, orientation: float, window: int) -> pd.Series:
    feature = series.astype(float).replace([np.inf, -np.inf], np.nan)
    rolling_mean = feature.rolling(window, min_periods=max(5, window // 2)).mean()
    rolling_std = feature.rolling(window, min_periods=max(5, window // 2)).std()
    zscore = (feature - rolling_mean) / (rolling_std + 1e-12)
    return (orientation * zscore).clip(-3.0, 3.0)


def _factor_weight(row: pd.Series, weighting: str) -> float:
    if weighting == "equal":
        return 1.0
    if weighting == "ic":
        train_ic = row.get("train_ic", float("nan"))
        try:
            value = abs(float(train_ic))
        except (TypeError, ValueError):
            value = float("nan")
        return float(value) if math.isfinite(value) and value > 1e-6 else 1e-6
    raise ValueError(f"unknown factor weighting: {weighting}")


def _build_symbol_scores(
    config: PortfolioBacktestConfig,
    kept: pd.DataFrame,
    *,
    weighting: str,
) -> pd.DataFrame:
    equities = load_zip_archive(config.zip_path, limit=config.limit)
    kept_by_symbol = {symbol: group.copy() for symbol, group in kept.groupby("symbol")}
    rows: list[pd.DataFrame] = []
    for equity in equities:
        if equity.symbol not in kept_by_symbol:
            continue
        frame = equity.frame.reset_index(drop=True)
        cutoff = int(len(frame) * config.split_ratio)
        factors = factor_series_map(frame, bank="alpha360")
        next_return = (frame["close"].shift(-1) / frame["close"] - 1.0).clip(-0.3, 0.3)
        score_parts: list[pd.Series] = []
        weight_parts: list[float] = []
        for _, decision in kept_by_symbol[equity.symbol].iterrows():
            factor_name = str(decision["factor_name"])
            if factor_name not in factors:
                continue
            factor_series = factors[factor_name][1]
            train_ic = decision.get("train_ic", float("nan"))
            try:
                train_ic_value = float(train_ic)
            except (TypeError, ValueError):
                train_ic_value = float("nan")
            orientation = 1.0 if not math.isfinite(train_ic_value) or train_ic_value >= 0.0 else -1.0
            factor_weight = _factor_weight(decision, weighting)
            score_parts.append(_factor_signal(factor_series, orientation=orientation, window=config.signal_window) * factor_weight)
            weight_parts.append(factor_weight)
        if not score_parts:
            continue
        symbol_score = sum(score_parts) / max(sum(weight_parts), 1e-12)
        symbol_rows = pd.DataFrame(
            {
                "date": frame["date"],
                "symbol": equity.symbol,
                "score": symbol_score,
                "next_return": next_return,
            }
        )
        symbol_rows = symbol_rows.iloc[cutoff:].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(symbol_rows)
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "score", "next_return"])
    return pd.concat(rows, axis=0, ignore_index=True)


def _daily_long_short(scores: pd.DataFrame, *, long_quantile: float, cost_bps: float) -> pd.DataFrame:
    daily_rows: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None
    for date, group in scores.groupby("date", sort=True):
        group = group.sort_values("score").dropna(subset=["score", "next_return"])
        n_assets = len(group)
        if n_assets < 2:
            continue
        side_count = max(1, int(np.floor(n_assets * long_quantile)))
        side_count = min(side_count, n_assets // 2)
        shorts = group.head(side_count)
        longs = group.tail(side_count)
        weights = pd.Series(0.0, index=group["symbol"].astype(str))
        weights.loc[longs["symbol"].astype(str)] = 0.5 / len(longs)
        weights.loc[shorts["symbol"].astype(str)] = -0.5 / len(shorts)
        returns = pd.Series(group["next_return"].to_numpy(dtype=float), index=group["symbol"].astype(str))
        gross_return = float((weights * returns).sum())
        if previous_weights is None:
            turnover = float(weights.abs().sum())
        else:
            aligned = pd.concat([weights, previous_weights], axis=1).fillna(0.0)
            turnover = float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().sum())
        cost = turnover * cost_bps / 10000.0
        daily_rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "net_return": gross_return - cost,
                "long_count": float(len(longs)),
                "short_count": float(len(shorts)),
                "n_assets": float(n_assets),
            }
        )
        previous_weights = weights
    return pd.DataFrame(daily_rows)


def run_portfolio_backtest(config: PortfolioBacktestConfig) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    kept = _load_kept_decisions(config)
    weightings = ["equal", "ic"] if config.factor_weighting == "both" else [config.factor_weighting]
    outputs: dict[str, pd.DataFrame] = {}
    summary: dict[str, float] = {
        "n_kept_factor_rows": float(len(kept)),
        "n_kept_symbols": float(kept["symbol"].nunique()) if not kept.empty else 0.0,
        "cost_bps": float(config.cost_bps),
        "long_quantile": float(config.long_quantile),
    }
    for weighting in weightings:
        scores = _build_symbol_scores(config, kept, weighting=weighting)
        daily = _daily_long_short(scores, long_quantile=config.long_quantile, cost_bps=config.cost_bps)
        outputs[f"{weighting}_scores"] = scores
        outputs[f"{weighting}_daily"] = daily
        summary.update(_summary_from_daily(daily, prefix=weighting))

    if config.output_dir is not None:
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        kept.to_csv(out / "kept_factors.csv", index=False)
        for name, frame in outputs.items():
            frame.to_csv(out / f"{name}.csv", index=False)
        (out / "portfolio_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return outputs, summary
