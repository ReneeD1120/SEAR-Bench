from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_assets: int = 120
    n_days: int = 720
    seed: int = 7
    regime_switch_day: int = 360
    horizon: int = 5


def generate_synthetic_benchmark(config: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rng = np.random.default_rng(config.seed)
    rows = []
    dates = pd.date_range("2020-01-01", periods=config.n_days, freq="B")
    truth = {
        "momentum_20d": {"label_keep": 1, "active_regime": "low_vol"},
        "reversal_5d": {"label_keep": 1, "active_regime": "high_vol"},
        "volume_surge": {"label_keep": 1, "active_regime": "high_vol"},
        "range_pct": {"label_keep": 0, "active_regime": "none"},
        "close_to_open": {"label_keep": 0, "active_regime": "none"},
        "noise_factor": {"label_keep": 0, "active_regime": "none"},
    }
    for asset in range(config.n_assets):
        price = float(100.0 + rng.normal())
        history_close: list[float] = [price]
        history_volume: list[float] = [float(rng.lognormal(mean=12, sigma=0.4))]
        for t, date in enumerate(dates):
            regime = 1 if t >= config.regime_switch_day else 0
            mom_20d = history_close[-1] / history_close[-21] - 1.0 if len(history_close) > 20 else 0.0
            reversal_5d = -(history_close[-1] / history_close[-6] - 1.0) if len(history_close) > 5 else 0.0
            volume_base = float(np.mean(history_volume[-20:]))
            vol_shock = float(np.clip(rng.normal(scale=0.12), -0.45, 0.45))
            volume = max(1.0, volume_base * (1.0 + 0.12 * regime) * np.exp(vol_shock))
            volume_surge = float(np.clip(np.exp(vol_shock) * (1.0 + 0.05 * regime), 0.5, 2.5))
            range_pct = float(np.clip(abs(rng.normal(loc=0.02 + 0.01 * regime, scale=0.008)), 0.002, 0.15))
            close_to_open = rng.normal(scale=0.003)
            noise_factor = rng.normal()

            if regime == 0:
                expected_ret = 0.035 * mom_20d + 0.010 * close_to_open
            else:
                expected_ret = 0.030 * volume_surge - 0.025 * reversal_5d
            noise = float(rng.normal(scale=0.012 + 0.006 * regime))
            log_ret = float(np.clip(expected_ret + noise, -0.06, 0.06))
            future_return = float(np.expm1(log_ret))

            open_ = float(price * np.exp(rng.normal(scale=0.004)))
            close = float(max(1.0, open_ * np.exp(log_ret)))
            high = float(max(open_, close) * (1 + abs(rng.normal(scale=range_pct * 0.25 + 0.003))))
            low = float(min(open_, close) * (1 - abs(rng.normal(scale=range_pct * 0.25 + 0.003))))
            rows.append({
                "asset": f"S{asset:04d}",
                "date": date,
                "open": open_,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "regime": regime,
                "future_return": future_return,
                "momentum_20d": mom_20d,
                "reversal_5d": reversal_5d,
                "volume_surge": volume_surge,
                "range_pct": range_pct,
                "close_to_open": close_to_open,
                "noise_factor": noise_factor,
            })
            history_close.append(close)
            history_volume.append(volume)
            price = close
    return pd.DataFrame(rows), truth
