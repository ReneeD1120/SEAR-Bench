from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["ret_5d"] = out["close"].pct_change(5)
    out["ret_10d"] = out["close"].pct_change(10)
    out["ret_20d"] = out["close"].pct_change(20)
    out["ret_60d"] = out["close"].pct_change(60)
    out["vol_20d"] = out["ret_1d"].rolling(20).std()
    out["vol_60d"] = out["ret_1d"].rolling(60).std()
    out["mom_20d"] = out["close"].pct_change(20)
    out["mom_60d"] = out["close"].pct_change(60)
    out["rev_5d"] = -out["close"].pct_change(5)
    out["rev_10d"] = -out["close"].pct_change(10)
    out["vol_surge"] = out["volume"] / out["volume"].rolling(20).mean()
    out["volume_ma_20"] = out["volume"].rolling(20).mean()
    out["volume_ma_60"] = out["volume"].rolling(60).mean()
    out["volume_ma_ratio_20"] = out["volume"] / out["volume_ma_20"]
    out["volume_ma_ratio_60"] = out["volume"] / out["volume_ma_60"]
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["hl_spread"] = (out["high"] - out["low"]) / out["open"]
    out["gap_return"] = out["open"] / out["close"].shift(1) - 1.0
    out["close_to_open"] = out["close"] / out["open"] - 1.0
    out["price_to_ma_20"] = out["close"] / out["close"].rolling(20).mean() - 1.0
    out["price_to_ma_60"] = out["close"] / out["close"].rolling(60).mean() - 1.0
    out["intraday_reversal"] = -(out["close"] / out["open"] - 1.0)
    return out


def make_candidate_factor_sets(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "momentum_20d": df["mom_20d"],
        "momentum_60d": df["mom_60d"],
        "reversal_5d": df["rev_5d"],
        "reversal_10d": df["rev_10d"],
        "volume_surge": df["vol_surge"],
        "volume_ma_ratio_20": df["volume_ma_ratio_20"],
        "volume_ma_ratio_60": df["volume_ma_ratio_60"],
        "range_pct": df["range_pct"],
        "hl_spread": df["hl_spread"],
        "close_to_open": df["close"] / df["open"] - 1.0,
        "gap_return": df["gap_return"],
        "ret_1d": df["ret_1d"],
        "ret_5d": df["ret_5d"],
        "ret_10d": df["ret_10d"],
        "ret_20d": df["ret_20d"],
        "ret_60d": df["ret_60d"],
        "vol_20d": df["vol_20d"],
        "vol_60d": df["vol_60d"],
        "price_to_ma_20": df["price_to_ma_20"],
        "price_to_ma_60": df["price_to_ma_60"],
        "intraday_reversal": df["intraday_reversal"],
    }


def future_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    return df["close"].shift(-horizon) / df["close"] - 1.0
