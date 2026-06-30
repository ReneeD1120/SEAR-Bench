from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_assets: int = 100
    n_days: int = 1000
    seed: int = 7
    regime_switch_day: int = 500


def generate_synthetic_market(config: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows = []
    dates = pd.date_range("2020-01-01", periods=config.n_days, freq="B")
    for asset in range(config.n_assets):
        price = 100.0 + rng.normal()
        prev_close = price
        for t, date in enumerate(dates):
            regime = 1 if t >= config.regime_switch_day else 0
            mom = 0.0 if t < 20 else price / max(1e-6, prev_close) - 1.0
            latent_signal = 0.08 * mom * (1 if regime == 0 else -1)
            noise = rng.normal(scale=0.02 + 0.01 * regime)
            ret = latent_signal + noise
            open_ = price * (1 + rng.normal(scale=0.005))
            close = max(1.0, open_ * (1 + ret))
            high = max(open_, close) * (1 + abs(rng.normal(scale=0.01)))
            low = min(open_, close) * (1 - abs(rng.normal(scale=0.01)))
            volume = max(1.0, rng.lognormal(mean=12, sigma=0.5) * (1 + 0.2 * regime))
            rows.append({
                "asset": f"S{asset:04d}",
                "date": date,
                "open": open_,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "regime": regime,
                "future_return": ret,
            })
            prev_close = close
            price = close
    return pd.DataFrame(rows)
