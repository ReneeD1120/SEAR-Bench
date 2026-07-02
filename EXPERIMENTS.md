# Experiment Notes

## Current Factor Pool

- `alpha158`: 52 Qlib-style OHLCV factors.
- `alpha360`: 276 total factors after AlphaBench-style expansion.
- Families include return, momentum, volatility, mean reversion, volume, volume volatility, range, gap, intraday, candlestick, normalization, lag, rank, min-max, spread, and interaction.

## Synthetic Benchmark

Command:

```bash
PYTHONPATH=src python -m seae.cli synthetic --n-assets 30 --n-days 720 --output-dir outputs/experiment_synthetic_30x720
```

Setup:

- 30 synthetic assets.
- 720 business days.
- Known regime switch.
- Known factor-validity labels.
- 2,493 held-out factor decisions.

Results:

- Learned judge keep accuracy: `0.6394`.
- Rule baseline keep accuracy: `0.5652`.
- Learned judge regime accuracy: `0.2768`.
- Kept factor mean test Sharpe: `0.1969`.
- Dropped factor mean test Sharpe: `-0.3364`.

Interpretation:

- The structured evidence scorer improves keep/drop classification relative to the deterministic rule baseline.
- Kept factors have better out-of-sample trading evidence than dropped factors.
- Regime identification is still weak and should be treated as an open research problem.

## Real Market Benchmark

Command:

```bash
PYTHONPATH=src python -m seae.cli real --zip-path /Users/renee/Downloads/RAFPO/不复权.zip --limit 20 --output-dir outputs/experiment_real_limit20
```

Setup:

- 20 U.S. equity files from the provided unadjusted OHLCV zip.
- 276 factors per symbol.
- 5,520 factor rows.
- Rule-based evidence judge only, because real labels are unknown.

Results:

- Average train IC: `-0.0452`.
- Average test IC: `-0.0570`.
- Keep rate: `0.1132`.
- Kept factor mean test Sharpe: `1.1501`.
- Dropped factor mean test Sharpe: `0.4145`.

Interpretation:

- The current rule baseline selects a small subset with stronger strategy proxy Sharpe.
- It does not select positive-IC factors on this unadjusted sample; kept average test IC is still negative.
- Mean reversion and momentum-style families dominate the real-data family ablation under the strategy proxy.
- Because the data are unadjusted, the real-market numbers should be treated as a pipeline validation rather than final finance evidence.

## Research Feedback

- The expanded factor pool is large enough to support downstream agentic reasoning experiments.
- The current code is an offline benchmark, not reinforcement learning.
- The next step is to let an LLM/agent consume only `sear reason` structured evidence, produce keep/drop/regime rationales, and evaluate those decisions on held-out benchmark metrics.
- A later version should add portfolio-level backtesting with adjusted prices, transaction costs, and cross-sectional long-short construction.
