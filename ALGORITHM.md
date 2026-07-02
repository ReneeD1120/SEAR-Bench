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
   - `alpha158` bank: Qlib-style core templates over returns, momentum, moving-average deviation, volatility, volume ratios, range, gap, intraday behavior, and candlestick geometry
   - `alpha360` bank: alpha158 plus AlphaBench-style expansions with lags, z-scores, rolling ranks, min-max normalization, spreads, and interactions
   - each factor carries a `family` field for downstream ablation and agent reasoning
3. Extract structured evidence for each factor:
   - `IC`
   - `ICIR`
   - `win rate`
   - `stability`
   - regime IC under high-vol and low-vol slices
   - regime contrast
   - train/test IC split
   - directional long-short proxy return
   - strategy Sharpe
   - cumulative return
   - max drawdown
   - The strategy return is a horizon-adjusted daily proxy: `sign(oriented factor signal) * clipped future return / horizon`.
4. Judge the factor:
   - `rule_based_judge` provides a deterministic baseline.
   - `LinearEvidenceJudge` learns a lightweight scorer from synthetic labels.
   - Both judges only consume in-sample structured evidence.
5. Build an agent reasoning view:
   - family-level summaries rank the most promising factor families
   - top factor candidates provide concrete evidence for agentic reasoning
   - the `sear reason` command emits this view as JSON for a downstream LLM agent
6. Run LLM reasoning:
   - `sear llm` sends the structured evidence view to an OpenAI-compatible chat endpoint.
   - `sear llm --backend hf-local` can also call a Hugging Face `transformers` model directly.
   - Qwen is the first open-source target model.
   - The LLM must return strict JSON keep/drop, active regime, confidence, and rationale fields.
   - The LLM still cannot inspect raw prices or hidden labels.
7. Evaluate:
   - synthetic: keep accuracy, regime accuracy, mean test IC of kept vs dropped factors, and mean test strategy Sharpe of kept vs dropped factors
   - real market: average train/test IC, keep rate, family ablation, strategy Sharpe, cumulative return, and drawdown
   - LLM: kept vs dropped test IC/Sharpe, rule agreement, and synthetic label accuracy when labels exist

## Outputs

- `synthetic_factor_table.csv`
- `synthetic_predictions.csv`
- `synthetic_summary.json`
- `synthetic_family_ablation.csv`
- `real_market_factor_table.csv`
- `real_market_family_ablation.csv`
- `real_market_summary.json`

## Command Line

```bash
sear synthetic --output-dir outputs
sear real --zip-path /Users/renee/Downloads/RAFPO/不复权.zip --limit 10 --output-dir outputs
sear reason --zip-path /Users/renee/Downloads/RAFPO/不复权.zip --limit 10 --top-k 5
sear llm --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 3 --top-k 3 --model Qwen/Qwen3-8B --dry-run
sear llm --backend hf-local --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 3 --top-k 3 --model Qwen/Qwen3-8B
```

## Interpretation

- If synthetic keep accuracy is high, the judge can recover known factor validity.
- If regime accuracy is high, the judge can infer the active market regime.
- If kept factors have higher out-of-sample strategy Sharpe than dropped factors, the reasoning layer is selecting economically useful candidates.
- On real data, the benchmark checks whether the scoring pipeline produces stable factor rankings, sensible train/test IC separation, and family-level return evidence.

## Current Limitation

The current implementation is an offline benchmark, not reinforcement learning. It does not yet feed realized classification or trading reward back into factor generation. The next research step is to run Qwen or another open-source LLM through `sear llm`, score its keep/drop/regime hypotheses on the held-out benchmark, and iterate under a strict no-raw-price-access protocol.
