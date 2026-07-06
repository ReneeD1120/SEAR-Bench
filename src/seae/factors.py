from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    formula: str
    transform: Callable[[pd.DataFrame], pd.Series]


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-12
    out["ret_1d"] = out["close"].pct_change()
    out["ret_2d"] = out["close"].pct_change(2)
    out["ret_5d"] = out["close"].pct_change(5)
    out["ret_10d"] = out["close"].pct_change(10)
    out["ret_20d"] = out["close"].pct_change(20)
    out["ret_60d"] = out["close"].pct_change(60)
    out["ret_120d"] = out["close"].pct_change(120)
    out["mom_5d"] = out["close"].pct_change(5)
    out["mom_10d"] = out["close"].pct_change(10)
    out["mom_20d"] = out["close"].pct_change(20)
    out["mom_60d"] = out["close"].pct_change(60)
    out["mom_120d"] = out["close"].pct_change(120)
    out["vol_5d"] = out["ret_1d"].rolling(5).std()
    out["vol_10d"] = out["ret_1d"].rolling(10).std()
    out["vol_20d"] = out["ret_1d"].rolling(20).std()
    out["vol_60d"] = out["ret_1d"].rolling(60).std()
    out["vol_120d"] = out["ret_1d"].rolling(120).std()
    out["volume_std_5"] = out["volume"].pct_change().rolling(5).std()
    out["volume_std_10"] = out["volume"].pct_change().rolling(10).std()
    out["volume_std_20"] = out["volume"].pct_change().rolling(20).std()
    out["volume_std_60"] = out["volume"].pct_change().rolling(60).std()
    out["volume_std_120"] = out["volume"].pct_change().rolling(120).std()
    out["close_ma_5"] = out["close"].rolling(5).mean()
    out["close_ma_10"] = out["close"].rolling(10).mean()
    out["close_ma_20"] = out["close"].rolling(20).mean()
    out["close_ma_60"] = out["close"].rolling(60).mean()
    out["close_ma_120"] = out["close"].rolling(120).mean()
    out["volume_ma_5"] = out["volume"].rolling(5).mean()
    out["volume_ma_10"] = out["volume"].rolling(10).mean()
    out["volume_ma_20"] = out["volume"].rolling(20).mean()
    out["volume_ma_60"] = out["volume"].rolling(60).mean()
    out["volume_ma_120"] = out["volume"].rolling(120).mean()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["hl_spread"] = (out["high"] - out["low"]) / out["open"]
    out["open_close_gap"] = out["open"] / out["close"].shift(1) - 1.0
    out["close_open_return"] = out["close"] / out["open"] - 1.0
    out["open_to_close"] = out["open"] / out["close"] - 1.0
    out["high_to_close"] = out["high"] / out["close"] - 1.0
    out["low_to_close"] = out["low"] / out["close"] - 1.0
    out["price_to_ma_5"] = out["close"] / out["close_ma_5"] - 1.0
    out["price_to_ma_10"] = out["close"] / out["close_ma_10"] - 1.0
    out["price_to_ma_20"] = out["close"] / out["close_ma_20"] - 1.0
    out["price_to_ma_60"] = out["close"] / out["close_ma_60"] - 1.0
    out["price_to_ma_120"] = out["close"] / out["close_ma_120"] - 1.0
    out["volume_ratio_5"] = out["volume"] / out["volume_ma_5"]
    out["volume_ratio_10"] = out["volume"] / out["volume_ma_10"]
    out["volume_ratio_20"] = out["volume"] / out["volume_ma_20"]
    out["volume_ratio_60"] = out["volume"] / out["volume_ma_60"]
    out["volume_ratio_120"] = out["volume"] / out["volume_ma_120"]
    out["intraday_reversal"] = -(out["close"] / out["open"] - 1.0)
    out["amplitude"] = (out["high"] - out["low"]) / out["open"]
    candle_range = out["high"] - out["low"]
    body_top = np.maximum(out["open"], out["close"])
    body_bottom = np.minimum(out["open"], out["close"])
    out["k_mid"] = (out["close"] - out["open"]) / (out["open"] + eps)
    out["k_mid2"] = (out["close"] - out["open"]) / (candle_range + eps)
    out["k_len"] = candle_range / (out["open"] + eps)
    out["k_upper"] = (out["high"] - body_top) / (out["open"] + eps)
    out["k_upper2"] = (out["high"] - body_top) / (candle_range + eps)
    out["k_lower"] = (body_bottom - out["low"]) / (out["open"] + eps)
    out["k_lower2"] = (body_bottom - out["low"]) / (candle_range + eps)
    out["k_shift"] = (2.0 * out["close"] - out["high"] - out["low"]) / (out["open"] + eps)
    out["k_shift2"] = (2.0 * out["close"] - out["high"] - out["low"]) / (candle_range + eps)
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
        return _safe_series(df[series].rolling(window).rank(pct=True))

    return transform


def _rolling_minmax(series: str, window: int) -> Callable[[pd.DataFrame], pd.Series]:
    def transform(df: pd.DataFrame) -> pd.Series:
        x = df[series]
        lo = x.rolling(window).min()
        hi = x.rolling(window).max()
        return _safe_series((x - lo) / (hi - lo + 1e-12))

    return transform


def _rolling_mean_factor(series: str, window: int) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: _safe_series(df[series].rolling(window).mean())


def _rolling_std_factor(series: str, window: int) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: _safe_series(df[series].rolling(window).std())


def _product(a: str, b: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: _safe_series(df[a] * df[b])


def alpha158_specs() -> list[FactorSpec]:
    specs: list[FactorSpec] = []
    for w in (5, 10, 20, 60, 120):
        specs.extend(
            [
                FactorSpec(f"ret_{w}d", "return", f"close / close.shift({w}) - 1", lambda df, w=w: df[f"ret_{w}d"]),
                FactorSpec(f"mom_{w}d", "momentum", f"close / close.shift({w}) - 1", lambda df, w=w: df[f"mom_{w}d"]),
                FactorSpec(f"vol_{w}d", "volatility", f"rolling_std(close.pct_change(), {w})", lambda df, w=w: df[f"vol_{w}d"]),
                FactorSpec(f"price_to_ma_{w}", "mean_reversion", f"close / rolling_mean(close, {w}) - 1", lambda df, w=w: df[f"price_to_ma_{w}"]),
                FactorSpec(f"volume_ratio_{w}", "volume", f"volume / rolling_mean(volume, {w})", lambda df, w=w: df[f"volume_ratio_{w}"]),
                FactorSpec(f"volume_std_{w}", "volume_volatility", f"rolling_std(volume.pct_change(), {w})", lambda df, w=w: df[f"volume_std_{w}"]),
            ]
        )
    specs.extend(
        [
            FactorSpec("range_pct", "range", "(high - low) / close", lambda df: df["range_pct"]),
            FactorSpec("hl_spread", "range", "(high - low) / open", lambda df: df["hl_spread"]),
            FactorSpec("open_close_gap", "gap", "open / close.shift(1) - 1", lambda df: df["open_close_gap"]),
            FactorSpec("close_open_return", "intraday", "close / open - 1", lambda df: df["close_open_return"]),
            FactorSpec("intraday_reversal", "intraday", "-(close / open - 1)", lambda df: df["intraday_reversal"]),
            FactorSpec("open_to_close", "price_level", "open / close - 1", lambda df: df["open_to_close"]),
            FactorSpec("high_to_close", "price_level", "high / close - 1", lambda df: df["high_to_close"]),
            FactorSpec("low_to_close", "price_level", "low / close - 1", lambda df: df["low_to_close"]),
            FactorSpec("k_mid", "candlestick", "(close - open) / open", lambda df: df["k_mid"]),
            FactorSpec("k_mid2", "candlestick", "(close - open) / (high - low)", lambda df: df["k_mid2"]),
            FactorSpec("k_len", "candlestick", "(high - low) / open", lambda df: df["k_len"]),
            FactorSpec("k_upper", "candlestick", "(high - max(open, close)) / open", lambda df: df["k_upper"]),
            FactorSpec("k_upper2", "candlestick", "(high - max(open, close)) / (high - low)", lambda df: df["k_upper2"]),
            FactorSpec("k_lower", "candlestick", "(min(open, close) - low) / open", lambda df: df["k_lower"]),
            FactorSpec("k_lower2", "candlestick", "(min(open, close) - low) / (high - low)", lambda df: df["k_lower2"]),
            FactorSpec("k_shift", "candlestick", "(2 * close - high - low) / open", lambda df: df["k_shift"]),
            FactorSpec("k_shift2", "candlestick", "(2 * close - high - low) / (high - low)", lambda df: df["k_shift2"]),
            FactorSpec("price_to_ma_5_z", "normalization", "zscore(price_to_ma_5, 20)", _rolling_zscore("price_to_ma_5", 20)),
            FactorSpec("price_to_ma_20_z", "normalization", "zscore(price_to_ma_20, 20)", _rolling_zscore("price_to_ma_20", 20)),
            FactorSpec("volume_ratio_20_z", "normalization", "zscore(volume_ratio_20, 20)", _rolling_zscore("volume_ratio_20", 20)),
            FactorSpec("amplitude_rank_20", "rank", "rolling_rank(amplitude, 20)", _rolling_rank("amplitude", 20)),
            FactorSpec("range_minmax_20", "rank", "rolling_minmax(range_pct, 20)", _rolling_minmax("range_pct", 20)),
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
        ("volume_std_20", "volume_volatility"),
        ("volume_ratio_20", "volume"),
        ("range_pct", "range"),
        ("price_to_ma_20", "mean_reversion"),
        ("open_close_gap", "gap"),
        ("k_mid", "candlestick"),
        ("k_upper", "candlestick"),
        ("k_lower", "candlestick"),
    ]
    windows = [5, 10, 20, 60]
    for series_name, family in base:
        for w in windows:
            specs.append(FactorSpec(f"{series_name}_z{w}", f"{family}_zscore", f"zscore({series_name}, {w})", _rolling_zscore(series_name, w)))
            specs.append(FactorSpec(f"{series_name}_rank{w}", f"{family}_rank", f"rolling_rank({series_name}, {w})", _rolling_rank(series_name, w)))
            specs.append(FactorSpec(f"{series_name}_minmax{w}", f"{family}_minmax", f"rolling_minmax({series_name}, {w})", _rolling_minmax(series_name, w)))
        for lag in (1, 2, 5):
            specs.append(FactorSpec(f"{series_name}_lag{lag}", f"{family}_lag", f"{series_name}.shift({lag})", _lagged(series_name, lag)))
    specs.extend(
        [
            FactorSpec("price_to_ma_5_minus_20", "spread", "price_to_ma_5 - price_to_ma_20", _diff("price_to_ma_5", "price_to_ma_20")),
            FactorSpec("price_to_ma_20_minus_60", "spread", "price_to_ma_20 - price_to_ma_60", _diff("price_to_ma_20", "price_to_ma_60")),
            FactorSpec("price_to_ma_60_minus_120", "spread", "price_to_ma_60 - price_to_ma_120", _diff("price_to_ma_60", "price_to_ma_120")),
            FactorSpec("volume_ratio_5_minus_20", "spread", "volume_ratio_5 - volume_ratio_20", _diff("volume_ratio_5", "volume_ratio_20")),
            FactorSpec("volume_ratio_20_minus_60", "spread", "volume_ratio_20 - volume_ratio_60", _diff("volume_ratio_20", "volume_ratio_60")),
            FactorSpec("mom_5_over_vol_20", "interaction", "mom_5d / vol_20d", _ratio("mom_5d", "vol_20d")),
            FactorSpec("mom_20_over_vol_60", "interaction", "mom_20d / vol_60d", _ratio("mom_20d", "vol_60d")),
            FactorSpec("ret_20_over_vol_60", "interaction", "ret_20d / vol_60d", _ratio("ret_20d", "vol_60d")),
            FactorSpec("range_over_volume", "interaction", "range_pct / volume_ratio_20", _ratio("range_pct", "volume_ratio_20")),
            FactorSpec("amplitude_over_vol", "interaction", "amplitude / vol_20d", _ratio("amplitude", "vol_20d")),
            FactorSpec("close_open_over_gap", "interaction", "close_open_return / open_close_gap", _ratio("close_open_return", "open_close_gap")),
            FactorSpec("k_mid_over_k_len", "interaction", "k_mid / k_len", _ratio("k_mid", "k_len")),
            FactorSpec("upper_minus_lower_shadow", "candlestick_spread", "k_upper - k_lower", _diff("k_upper", "k_lower")),
            FactorSpec("upper2_minus_lower2_shadow", "candlestick_spread", "k_upper2 - k_lower2", _diff("k_upper2", "k_lower2")),
        ]
    )
    return specs


def broad_expansion_specs() -> list[FactorSpec]:
    """Larger readable factor pool for downstream agentic reasoning.

    This keeps the generator Qlib/AlphaBench-like: start from OHLCV-derived
    primitives, apply rolling transforms/lags, then add interpretable spreads
    and interactions. Every generated factor still has a formula string.
    """
    specs: list[FactorSpec] = []
    base = [
        ("ret_1d", "return"), ("ret_2d", "return"), ("ret_5d", "return"), ("ret_10d", "return"), ("ret_20d", "return"), ("ret_60d", "return"), ("ret_120d", "return"),
        ("mom_5d", "momentum"), ("mom_10d", "momentum"), ("mom_20d", "momentum"), ("mom_60d", "momentum"), ("mom_120d", "momentum"),
        ("vol_5d", "volatility"), ("vol_10d", "volatility"), ("vol_20d", "volatility"), ("vol_60d", "volatility"), ("vol_120d", "volatility"),
        ("volume_std_5", "volume_volatility"), ("volume_std_10", "volume_volatility"), ("volume_std_20", "volume_volatility"), ("volume_std_60", "volume_volatility"), ("volume_std_120", "volume_volatility"),
        ("volume_ratio_5", "volume"), ("volume_ratio_10", "volume"), ("volume_ratio_20", "volume"), ("volume_ratio_60", "volume"), ("volume_ratio_120", "volume"),
        ("price_to_ma_5", "mean_reversion"), ("price_to_ma_10", "mean_reversion"), ("price_to_ma_20", "mean_reversion"), ("price_to_ma_60", "mean_reversion"), ("price_to_ma_120", "mean_reversion"),
        ("range_pct", "range"),
        ("hl_spread", "range"),
        ("open_close_gap", "gap"),
        ("close_open_return", "intraday"),
        ("intraday_reversal", "intraday"),
        ("open_to_close", "price_level"),
        ("high_to_close", "price_level"),
        ("low_to_close", "price_level"),
        ("amplitude", "range"),
        ("k_mid", "candlestick"),
        ("k_mid2", "candlestick"),
        ("k_len", "candlestick"),
        ("k_upper", "candlestick"),
        ("k_upper2", "candlestick"),
        ("k_lower", "candlestick"),
        ("k_lower2", "candlestick"),
        ("k_shift", "candlestick"),
        ("k_shift2", "candlestick"),
    ]
    windows = [5, 10, 20, 60, 120]
    for series_name, family in base:
        for w in windows:
            specs.extend(
                [
                    FactorSpec(f"{series_name}_mean{w}", f"{family}_mean", f"rolling_mean({series_name}, {w})", _rolling_mean_factor(series_name, w)),
                    FactorSpec(f"{series_name}_std{w}", f"{family}_std", f"rolling_std({series_name}, {w})", _rolling_std_factor(series_name, w)),
                    FactorSpec(f"{series_name}_zwide{w}", f"{family}_zscore", f"zscore({series_name}, {w})", _rolling_zscore(series_name, w)),
                    FactorSpec(f"{series_name}_rankwide{w}", f"{family}_rank", f"rolling_rank({series_name}, {w})", _rolling_rank(series_name, w)),
                    FactorSpec(f"{series_name}_minmaxwide{w}", f"{family}_minmax", f"rolling_minmax({series_name}, {w})", _rolling_minmax(series_name, w)),
                ]
            )
        for lag in (1, 2, 5, 10):
            specs.append(FactorSpec(f"{series_name}_lagwide{lag}", f"{family}_lag", f"{series_name}.shift({lag})", _lagged(series_name, lag)))

    for short, long in ((5, 10), (5, 20), (10, 20), (20, 60), (60, 120)):
        specs.extend(
            [
                FactorSpec(f"ret_{short}d_minus_{long}d_wide", "return_spread", f"ret_{short}d - ret_{long}d", _diff(f"ret_{short}d", f"ret_{long}d")),
                FactorSpec(f"mom_{short}d_minus_{long}d_wide", "momentum_spread", f"mom_{short}d - mom_{long}d", _diff(f"mom_{short}d", f"mom_{long}d")),
                FactorSpec(f"vol_{short}d_minus_{long}d_wide", "volatility_spread", f"vol_{short}d - vol_{long}d", _diff(f"vol_{short}d", f"vol_{long}d")),
                FactorSpec(f"volume_ratio_{short}_minus_{long}_wide", "volume_spread", f"volume_ratio_{short} - volume_ratio_{long}", _diff(f"volume_ratio_{short}", f"volume_ratio_{long}")),
                FactorSpec(f"price_to_ma_{short}_minus_{long}_wide", "ma_spread", f"price_to_ma_{short} - price_to_ma_{long}", _diff(f"price_to_ma_{short}", f"price_to_ma_{long}")),
            ]
        )

    for mom_w, vol_w in ((5, 10), (5, 20), (10, 20), (20, 60), (60, 120)):
        specs.extend(
            [
                FactorSpec(f"mom_{mom_w}d_over_vol_{vol_w}d_wide", "risk_adjusted_momentum", f"mom_{mom_w}d / vol_{vol_w}d", _ratio(f"mom_{mom_w}d", f"vol_{vol_w}d")),
                FactorSpec(f"ret_{mom_w}d_over_vol_{vol_w}d_wide", "risk_adjusted_return", f"ret_{mom_w}d / vol_{vol_w}d", _ratio(f"ret_{mom_w}d", f"vol_{vol_w}d")),
                FactorSpec(f"mom_{mom_w}d_x_volume_ratio_{vol_w}", "price_volume_interaction", f"mom_{mom_w}d * volume_ratio_{vol_w}", _product(f"mom_{mom_w}d", f"volume_ratio_{vol_w}")),
            ]
        )

    specs.extend(
        [
            FactorSpec("range_x_volume_ratio_20", "price_volume_interaction", "range_pct * volume_ratio_20", _product("range_pct", "volume_ratio_20")),
            FactorSpec("gap_x_volume_ratio_20", "price_volume_interaction", "open_close_gap * volume_ratio_20", _product("open_close_gap", "volume_ratio_20")),
            FactorSpec("intraday_x_volume_ratio_20", "price_volume_interaction", "close_open_return * volume_ratio_20", _product("close_open_return", "volume_ratio_20")),
            FactorSpec("k_mid_x_volume_ratio_20", "candlestick_volume_interaction", "k_mid * volume_ratio_20", _product("k_mid", "volume_ratio_20")),
            FactorSpec("k_upper_minus_k_lower_wide", "candlestick_spread", "k_upper - k_lower", _diff("k_upper", "k_lower")),
            FactorSpec("k_shift_minus_k_mid", "candlestick_spread", "k_shift - k_mid", _diff("k_shift", "k_mid")),
        ]
    )
    return specs


def _specs_for_bank(bank: str) -> list[FactorSpec]:
    if bank == "alpha158":
        specs = alpha158_specs()
    elif bank == "alpha360":
        specs = alpha158_specs() + expansion_specs()
    elif bank == "alpha1000":
        specs = alpha158_specs() + expansion_specs() + broad_expansion_specs()
    else:
        raise ValueError(f"unknown factor bank: {bank}")

    deduped: dict[str, FactorSpec] = {}
    for spec in specs:
        if spec.name in deduped:
            raise ValueError(f"duplicate factor name in {bank}: {spec.name}")
        deduped[spec.name] = spec
    return list(deduped.values())


def build_factor_bank(df: pd.DataFrame, *, bank: str = "alpha158") -> pd.DataFrame:
    out = add_basic_features(df)
    specs = _specs_for_bank(bank)
    frames = []
    for spec in specs:
        s = _safe_series(spec.transform(out))
        frames.append(
            pd.DataFrame(
                {
                    "factor_name": spec.name,
                    "family": spec.family,
                    "formula": spec.formula,
                    "value": s.astype(float),
                },
                index=out.index,
            )
        )
    bank_df = pd.concat(frames, axis=0, ignore_index=True)
    return bank_df


def factor_series_map(df: pd.DataFrame, *, bank: str = "alpha158") -> dict[str, tuple[str, pd.Series]]:
    out = add_basic_features(df)
    specs = _specs_for_bank(bank)
    result: dict[str, tuple[str, pd.Series]] = {}
    for spec in specs:
        result[spec.name] = (spec.family, _safe_series(spec.transform(out)))
    return result


def factor_metadata_map(*, bank: str = "alpha158") -> dict[str, FactorSpec]:
    return {spec.name: spec for spec in _specs_for_bank(bank)}


def make_candidate_factor_sets(df: pd.DataFrame, *, bank: str = "alpha158") -> dict[str, pd.Series]:
    return {name: series for name, (_, series) in factor_series_map(df, bank=bank).items()}


add_basic_factors = add_basic_features
