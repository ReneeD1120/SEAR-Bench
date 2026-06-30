from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["ret_5d"] = out["close"].pct_change(5)
    out["ret_20d"] = out["close"].pct_change(20)
    out["vol_20d"] = out["ret_1d"].rolling(20).std()
    out["mom_20d"] = out["close"].pct_change(20)
    out["rev_5d"] = -out["close"].pct_change(5)
    out["vol_surge"] = out["volume"] / out["volume"].rolling(20).mean()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    return out


def make_candidate_factor_sets(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "momentum_20d": df["mom_20d"],
        "reversal_5d": df["rev_5d"],
        "vol_surge": df["vol_surge"],
        "range_pct": df["range_pct"],
        "close_to_open": df["close"] / df["open"] - 1.0,
    }


def future_return(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    return df["close"].shift(-horizon) / df["close"] - 1.0

