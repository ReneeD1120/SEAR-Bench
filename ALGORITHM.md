# SEAR-Bench Algorithm

## Goal

SEAR-Bench studies whether a structured-evidence agent can judge factor validity and regime dependence from statistical summaries rather than raw text.

## Inputs

- A universe of U.S. equity OHLCV time series from the provided CSV zip archive.
- A synthetic benchmark with known regime labels and known factor validity.
- A library of candidate factors built from daily price and volume data.

## Pipeline

1. Load each asset time series.
2. Build candidate factors:
   - `momentum_20d`
   - `reversal_5d`
   - `volume_surge`
   - `range_pct`
   - `close_to_open`
   - `noise_factor` for synthetic data only
3. Extract structured evidence for each factor:
   - `IC`
   - `ICIR`
   - `win rate`
   - `stability`
   - regime IC under high-vol and low-vol slices
   - regime contrast
   - train/test IC split
   - factor templates inspired by `Alpha158` / Qlib-style return, moving-average, volatility, range, and volume transforms
4. Judge the factor:
   - `rule_based_judge` provides a deterministic baseline.
   - `LinearEvidenceJudge` learns a lightweight scorer from synthetic labels.
5. Evaluate:
   - synthetic: keep accuracy, regime accuracy, mean test IC of kept vs dropped factors
   - real market: average train/test IC, keep rate, and ranked factor outputs

## Outputs

- `synthetic_factor_table.csv`
- `synthetic_predictions.csv`
- `synthetic_summary.json`
- `real_market_factor_table.csv`
- `real_market_summary.json`

## Command Line

```bash
sear synthetic --output-dir outputs
sear real --zip-path /Users/renee/Downloads/RAFPO/不复权.zip --limit 10 --output-dir outputs
```

## Interpretation

- If synthetic keep accuracy is high, the judge can recover known factor validity.
- If regime accuracy is high, the judge can infer the active market regime.
- On real data, the benchmark checks whether the scoring pipeline produces stable factor rankings and sensible train/test IC separation.
