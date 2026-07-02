from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    transform: Callable[[pd.DataFrame], pd.Series]


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["ret_2d"] = out["close"].pct_change(2)
    out["ret_5d"] = out["close"].pct_change(5)
    out["ret_10d"] = out["close"].pct_change(10)
    out["ret_20d"] = out["close"].pct_change(20)
    out["ret_60d"] = out["close"].pct_change(60)
    out["mom_5d"] = out["close"].pct_change(5)
    out["mom_10d"] = out["close"].pct_change(10)
    out["mom_20d"] = out["close"].pct_change(20)
    out["mom_60d"] = out["close"].pct_change(60)
    out["vol_5d"] = out["ret_1d"].rolling(5).std()
    out["vol_10d"] = out["ret_1d"].rolling(10).std()
    out["vol_20d"] = out["ret_1d"].rolling(20).std()
    out["vol_60d"] = out["ret_1d"].rolling(60).std()
    out["close_ma_5"] = out["close"].rolling(5).mean()
    out["close_ma_10"] = out["close"].rolling(10).mean()
    out["close_ma_20"] = out["close"].rolling(20).mean()
    out["close_ma_60"] = out["close"].rolling(60).mean()
    out["volume_ma_5"] = out["volume"].rolling(5).mean()
    out["volume_ma_10"] = out["volume"].rolling(10).mean()
    out["volume_ma_20"] = out["volume"].rolling(20).mean()
    out["volume_ma_60"] = out["volume"].rolling(60).mean()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["hl_spread"] = (out["high"] - out["low"]) / out["open"]
    out["open_close_gap"] = out["open"] / out["close"].shift(1) - 1.0
    out["close_open_return"] = out["close"] / out["open"] - 1.0
    out["price_to_ma_5"] = out["close"] / out["close_ma_5"] - 1.0
    out["price_to_ma_10"] = out["close"] / out["close_ma_10"] - 1.0
    out["price_to_ma_20"] = out["close"] / out["close_ma_20"] - 1.0
    out["price_to_ma_60"] = out["close"] / out["close_ma_60"] - 1.0
    out["volume_ratio_5"] = out["volume"] / out["volume_ma_5"]
    out["volume_ratio_10"] = out["volume"] / out["volume_ma_10"]
    out["volume_ratio_20"] = out["volume"] / out["volume_ma_20"]
    out["volume_ratio_60"] = out["volume"] / out["volume_ma_60"]
    out["intraday_reversal"] = -(out["close"] / out["open"] - 1.0)
    out["amplitude"] = (out["high"] - out["low"]) / out["open"]
    return out


def _safe_series(x: pd.Series) -> pd.Series:
    return x.replace([np.inf, -np.inf], np.nan)


def _lagged(series_name: str, lag: int = 1) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: _safe_series(df[series_name].shift(lag))


def _ratio(numer: str, denom: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: _safe_series(df[numer] / df[denom])


def _diff(a: str, b: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: _safe_series(df[a] - df[b])


def _rolling_zscore(series: str, window: int) -> Callable[[pd.DataFrame], pd.Series]:
    def transform(df: pd.DataFrame) -> pd.Series:
        x = df[series]
        mu = x.rolling(window).mean()
        sigma = x.rolling(window).std()
        return _safe_series((x - mu) / (sigma + 1e-12))

    return transform


def _rolling_rank(series: str, window: int) -> Callable[[pd.DataFrame], pd.Series]:
    def transform(df: pd.DataFrame) -> pd.Series:
        return _safe_series(df[series].rolling(window).apply(lambda s: pd.Series(s).rank().iloc[-1] / len(s) if len(s) else np.nan))

    return transform


def _rolling_minmax(series: str, window: int) -> Callable[[pd.DataFrame], pd.Series]:
    def transform(df: pd.DataFrame) -> pd.Series:
        x = df[series]
        lo = x.rolling(window).min()
        hi = x.rolling(window).max()
        return _safe_series((x - lo) / (hi - lo + 1e-12))

    return transform


def alpha158_specs() -> list[FactorSpec]:
    specs: list[FactorSpec] = []
    for w in (5, 10, 20, 60):
        specs.extend(
            [
                FactorSpec(f"ret_{w}d", "return", lambda df, w=w: df[f"ret_{w}d"]),
                FactorSpec(f"mom_{w}d", "momentum", lambda df, w=w: df[f"mom_{w}d"]),
                FactorSpec(f"vol_{w}d", "volatility", lambda df, w=w: df[f"vol_{w}d"]),
                FactorSpec(f"price_to_ma_{w}", "mean_reversion", lambda df, w=w: df[f"price_to_ma_{w}"]),
                FactorSpec(f"volume_ratio_{w}", "volume", lambda df, w=w: df[f"volume_ratio_{w}"]),
            ]
        )
    specs.extend(
        [
            FactorSpec("range_pct", "range", lambda df: df["range_pct"]),
            FactorSpec("hl_spread", "range", lambda df: df["hl_spread"]),
            FactorSpec("open_close_gap", "gap", lambda df: df["open_close_gap"]),
            FactorSpec("close_open_return", "intraday", lambda df: df["close_open_return"]),
            FactorSpec("intraday_reversal", "intraday", lambda df: df["intraday_reversal"]),
            FactorSpec("price_to_ma_5_z", "normalization", _rolling_zscore("price_to_ma_5", 20)),
            FactorSpec("price_to_ma_20_z", "normalization", _rolling_zscore("price_to_ma_20", 20)),
            FactorSpec("volume_ratio_20_z", "normalization", _rolling_zscore("volume_ratio_20", 20)),
            FactorSpec("amplitude_rank_20", "rank", _rolling_rank("amplitude", 20)),
            FactorSpec("range_minmax_20", "rank", _rolling_minmax("range_pct", 20)),
        ]
    )
    return specs


def expansion_specs() -> list[FactorSpec]:
    specs: list[FactorSpec] = []
    base = [
        ("ret_1d", "return"),
        ("ret_5d", "return"),
        ("ret_20d", "return"),
        ("mom_5d", "momentum"),
        ("mom_20d", "momentum"),
        ("vol_20d", "volatility"),
        ("volume_ratio_20", "volume"),
        ("range_pct", "range"),
        ("price_to_ma_20", "mean_reversion"),
        ("open_close_gap", "gap"),
    ]
    windows = [5, 10, 20, 60]
    for series_name, family in base:
        for w in windows:
            specs.append(FactorSpec(f"{series_name}_z{w}", f"{family}_zscore", _rolling_zscore(series_name, w)))
            specs.append(FactorSpec(f"{series_name}_lag1", f"{family}_lag", _lagged(series_name, 1)))
            specs.append(FactorSpec(f"{series_name}_lag2", f"{family}_lag", _lagged(series_name, 2)))
    specs.extend(
        [
            FactorSpec("price_to_ma_5_minus_20", "spread", _diff("price_to_ma_5", "price_to_ma_20")),
            FactorSpec("price_to_ma_20_minus_60", "spread", _diff("price_to_ma_20", "price_to_ma_60")),
            FactorSpec("volume_ratio_5_minus_20", "spread", _diff("volume_ratio_5", "volume_ratio_20")),
            FactorSpec("mom_5_over_vol_20", "interaction", _ratio("mom_5d", "vol_20d")),
            FactorSpec("mom_20_over_vol_60", "interaction", _ratio("mom_20d", "vol_60d")),
            FactorSpec("range_over_volume", "interaction", _ratio("range_pct", "volume_ratio_20")),
            FactorSpec("amplitude_over_vol", "interaction", _ratio("amplitude", "vol_20d")),
            FactorSpec("close_open_over_gap", "interaction", _ratio("close_open_return", "open_close_gap")),
        ]
    )
    return specs


def build_factor_bank(df: pd.DataFrame, *, bank: str = "alpha158") -> pd.DataFrame:
    out = add_basic_features(df)
    specs = alpha158_specs() if bank == "alpha158" else alpha158_specs() + expansion_specs()
    frames = []
    for spec in specs:
        s = spec.transform(out)
        frames.append(
            pd.DataFrame(
                {
                    "factor_name": spec.name,
                    "family": spec.family,
                    "value": s.astype(float),
                },
                index=out.index,
            )
        )
    bank_df = pd.concat(frames, axis=0, ignore_index=True)
    return bank_df


def factor_series_map(df: pd.DataFrame, *, bank: str = "alpha158") -> dict[str, tuple[str, pd.Series]]:
    out = add_basic_features(df)
    specs = alpha158_specs() if bank == "alpha158" else alpha158_specs() + expansion_specs()
    result: dict[str, tuple[str, pd.Series]] = {}
    for spec in specs:
        result[spec.name] = (spec.family, spec.transform(out))
    return result
