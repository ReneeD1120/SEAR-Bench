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
   - each factor carries a `name`, `family`, and human-readable `formula`
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
   - the formal view is family-blind by default: no `family` field and no `family_summary`
   - top factor candidates provide `factor_name`, `formula`, train-only `train_factor_sample`, and train-only statistical evidence
   - `--factor-sample-size` controls how many train-only factor values are sent to the LLM; use small values for local GPU models
   - the `sear reason` command emits this view as JSON for a downstream LLM agent
   - `sear reason` defaults to the leakage-free version used by `sear llm`
   - `sear reason --diagnostic-leaky` emits the held-out diagnostic view and must not be used as LLM input
6. Run LLM reasoning:
   - `sear llm` sends the structured evidence view to an OpenAI-compatible chat endpoint.
   - `sear llm --backend hf-local` can also call a Hugging Face `transformers` model directly.
   - Qwen is the first open-source target model.
   - The LLM must return strict JSON keep/drop, active regime, confidence, and an `evidence_audit`.
   - `evidence_audit` contains formula hypothesis, support summary, counter-evidence, regime summary, and decision logic for each candidate.
   - The LLM can inspect factor names, formulas, train-only factor samples, and train-only evidence.
   - The LLM still cannot inspect raw prices, hidden labels, rule decisions, or held-out test metrics.
   - `--include-evidence-tags` enables a tag-assisted ablation; it is not the default formal reasoning view.
   - `--include-family` and `--include-family-summary` enable family-assisted ablations; they are not the default formal reasoning view.
   - `--revision-rounds` enables critic-guided revision before held-out evaluation.
7. Evaluate:
   - synthetic: keep accuracy, keep precision/recall/balanced accuracy, regime accuracy, mean test IC of kept vs dropped factors, and mean test strategy Sharpe of kept vs dropped factors
   - real market: average train/test IC, keep rate, family ablation, strategy Sharpe, cumulative return, and drawdown
   - LLM: kept vs dropped test IC/Sharpe, rule agreement, synthetic label accuracy when labels exist, and reasoning-quality diagnostics
   - reasoning diagnostics include confidence diversity, non-uncertain regime rate, audit non-empty rate, and audit uniqueness rate
   - explanation faithfulness checks whether support/counter/regime/decision explanations are consistent with structured train evidence
   - portfolio backtest evaluates kept factors as a test-period cross-sectional long-short strategy with equal/IC weighting, transaction costs, turnover, Sharpe, cumulative return, and drawdown

## Strong Reasoning Target Structure

The current bounded audit is a single-pass agentic baseline. A stronger reasoning architecture should use separate stages:

1. Hypothesis generation: infer the economic meaning of each formula.
2. Evidence audit: summarize train-only support and counter-evidence.
3. Regime specialist: decide whether high-vol, low-vol, none, or uncertain is supported.
4. Critic: check whether support/counter/regime statements are faithful to the structured evidence.
5. Revision: force the model to repair contradicted explanations before final keep/drop.
6. Final judge: output decisions only after the critic passes.
7. Benchmark: score decisions on held-out IC, strategy Sharpe, portfolio backtest, and explanation faithfulness.

This keeps the LLM in the decision/explanation role while making reasoning quality measurable.

The implemented revision workflow is:

```bash
sear llm ... --revision-rounds 1 --critic-out outputs/critic.json
```

The critic is deterministic and train-only. It checks audit-field polarity, regime consistency, and decision-logic contradictions before asking the LLM to revise. Held-out metrics are joined only after the final response.

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
sear llm --backend hf-local --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 3 --top-k 3 --factor-sample-size 3 --model Qwen/Qwen3-8B
```

## Interpretation

- If synthetic keep accuracy is high, the judge can recover known factor validity.
- If regime accuracy is high, the judge can infer the active market regime.
- If kept factors have higher out-of-sample strategy Sharpe than dropped factors, the reasoning layer is selecting economically useful candidates.
- On real data, the benchmark checks whether the scoring pipeline produces stable factor rankings, sensible train/test IC separation, and family-level return evidence.
- Family-assisted results should be reported as ablations. The main LLM benchmark should be family-blind so the model cannot solve the task by memorizing factor-family priors.

## Current Limitation

The current implementation is an offline benchmark, not reinforcement learning. It does not yet feed realized classification or trading reward back into factor generation. The next research step is to run Qwen or another open-source LLM through `sear llm`, score its keep/drop/regime hypotheses on the held-out benchmark, and iterate under a strict no-raw-price-access protocol.
