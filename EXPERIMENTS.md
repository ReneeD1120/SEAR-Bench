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

## Forward-Adjusted Real Market Benchmark

Command:

```bash
PYTHONPATH=src python -m seae.cli real --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 20 --output-dir outputs/experiment_real_qfq_limit20
```

Setup:

- Same first 20 files and same 276-factor pool as the unadjusted benchmark.
- Forward-adjusted OHLCV data reduce corporate-action artifacts relative to unadjusted prices.

Results:

- Average train IC: `-0.0416`.
- Average test IC: `-0.0569`.
- Keep rate: `0.1219`.
- Kept factor mean test Sharpe: `0.9248`.
- Dropped factor mean test Sharpe: `0.3363`.

Comparison with unadjusted data:

- Kept Sharpe remains higher than dropped Sharpe, so the keep/drop pipeline still has economic signal under the adjusted-price version.
- The absolute Sharpe is lower than the unadjusted run, which is expected because adjustment removes some artificial corporate-action jumps.
- The kept average test IC remains negative, so the current real-data baseline still behaves more like a strategy-proxy selector than a positive-IC selector.
- The first 20 symbols include non-common-stock-like tickers such as units or rights; a stronger paper experiment should add common-stock, liquidity, and minimum-history filters.

## Research Feedback

- The expanded factor pool is large enough to support downstream agentic reasoning experiments.
- `sear llm` now provides the first real LLM reasoning interface.
- Qwen is the first open-source target model.
- Supported Qwen paths:
  - OpenAI-compatible endpoint, for vLLM/llama.cpp server/hosted compatible APIs.
  - Hugging Face local `transformers` backend via `sear llm --backend hf-local`.
- Dry-run command tested:

```bash
PYTHONPATH=src python -m seae.cli llm --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 3 --top-k 3 --model Qwen/Qwen3-8B --prompt-out outputs/qwen_prompt_smoke.json --dry-run
```

- Hugging Face dry-run syntax tested:

```bash
PYTHONPATH=src python -m seae.cli llm --backend hf-local --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 2 --top-k 2 --model Qwen/Qwen3-8B --prompt-out outputs/qwen_prompt_verify.json --dry-run
```

- Real Hugging Face smoke tested with `Qwen/Qwen2.5-0.5B-Instruct`:

```bash
HF_HOME=.cache/huggingface PYTHONPATH=src .venv-hf/bin/python -m seae.cli llm --backend hf-local --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 1 --top-k 1 --model Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 512 --prompt-out outputs/qwen_hf_prompt_candidate2.json --response-out outputs/qwen_hf_response_candidate2.json --decisions-out outputs/qwen_hf_decisions_candidate2.csv
```

- Result: the local Qwen call completed, but `Qwen2.5-0.5B-Instruct` returned invalid JSON, so the benchmark correctly recorded `parse_success=0.0` and `n_decisions=0.0`.
- Interpretation: the Hugging Face backend works, but the 0.5B model is too weak for reliable structured JSON reasoning. Next tests should use a stronger Qwen model or constrained JSON decoding.

## Remote GPU Qwen Smoke

Environment:

- Remote Windows host: `WIN-CMSMMA0PCTS`.
- GPU: 3 x NVIDIA Quadro RTX 6000, 24GB each.
- Conda env: `searbench`, Python 3.11.
- CUDA PyTorch: `torch 2.7.1+cu118`, `torch.cuda.is_available=True`.
- Data copied to remote as `C:\Users\drx\SEAR-Bench\data\qfq.zip`.

Windows compatibility fix:

- `NamedTemporaryFile`-based zip loading failed on Windows because pandas could not reopen the temporary CSV.
- `load_zip_archive` now reads CSV bytes directly from the zip archive.

Commands tested:

```bash
conda run -n searbench python -m seae.cli real --zip-path data\qfq.zip --limit 2 --output-dir outputs\remote_qfq_smoke
```

Result:

- Real data smoke passed with `552` factor rows.

Qwen 1.5B:

```bash
conda run -n searbench python -m seae.cli llm --backend hf-local --zip-path data\qfq.zip --limit 2 --top-k 2 --model Qwen/Qwen2.5-1.5B-Instruct --hf-device-map auto --max-new-tokens 1024
```

- `parse_success=1.0`.
- `n_decisions=6`, `n_matched=6`.
- `keep_rate=0.0`.
- `rule_agreement_rate=1.0`.
- Note: this run used the early diagnostic reasoning view and should be treated as a pipeline smoke, not a formal held-out LLM benchmark.

Qwen 3B:

```bash
conda run -n searbench python -m seae.cli llm --backend hf-local --zip-path data\qfq.zip --limit 2 --top-k 2 --model Qwen/Qwen2.5-3B-Instruct --hf-device-map auto --max-new-tokens 1024
```

- `parse_success=1.0`.
- `n_decisions=6`, `n_matched=6`.
- `keep_rate=0.1667`.
- `rule_agreement_rate=0.8333`.
- Kept mean test IC: `0.1235`; dropped mean test IC: `-0.0069`.
- Kept mean test Sharpe: `1.6180`; dropped mean test Sharpe: `1.8925`.
- Note: this run used the early diagnostic reasoning view and should be treated as a pipeline smoke, not a formal held-out LLM benchmark.

Interpretation:

- Remote GPU execution is now working.
- Qwen 3B is a usable first open-source LLM reasoning baseline: it returns valid JSON and exact candidate matches.
- The rationale is not fully faithful yet; it still makes occasional numerical interpretation errors. This suggests SEAR-Bench should evaluate both decision metrics and explanation faithfulness.
- Formal LLM evaluation now uses `build_llm_reasoning_view`, which exposes only train/in-sample evidence and hides all held-out test metrics until scoring.

- The current code is an offline benchmark, not reinforcement learning.
- The next step is to serve Qwen, let it consume only structured evidence, produce keep/drop/regime rationales, and evaluate those decisions on held-out benchmark metrics.
- A later version should add portfolio-level backtesting with adjusted prices, transaction costs, and cross-sectional long-short construction.

## Formal No-Leak Qwen Runs

Candidate selection fix:

- The formal LLM view now filters candidates to at least `252` train observations.
- Candidates are selected by train-only evidence and diversified across symbols/families.
- Held-out metrics remain hidden from the LLM and are joined only after decisions are produced.
- Evaluation summary now reports match and valid held-out strategy coverage.

Compact rationale-code prompt:

- Qwen 1.5B and 3B both parse successfully with compact reason codes.
- With raw numeric fields only, both models were overly conservative and dropped all 15 candidates.
- This was a useful negative result: small open-source Qwen models can produce valid JSON while still misreading structured numeric evidence.

Tagged train-evidence prompt:

- Added train-only `evidence_tags` for each candidate: `strategy_strength`, `ic_signal`, `history_quality`, `drawdown_risk`, and `regime_signal`.
- These tags do not expose held-out test metrics; they summarize only in-sample benchmark evidence.

Results on `qfq.zip`, `limit=5`, `top_k=5`, 15 candidates:

| Model | View | Parse | Match | Valid Sharpe | Keep Rate | Kept Test Sharpe | Dropped Test Sharpe | Rule Agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B | compact codes | 1.0 | 1.0 | 1.0 | 0.0000 | NaN | -0.1431 | 0.8667 |
| Qwen2.5-3B | compact codes | 1.0 | 1.0 | 1.0 | 0.0000 | NaN | -0.1431 | 0.8667 |
| Qwen2.5-1.5B | tagged evidence | 1.0 | 1.0 | 1.0 | 0.0000 | NaN | -0.1431 | 0.8667 |
| Qwen2.5-3B | tagged evidence | 1.0 | 1.0 | 1.0 | 0.3333 | 0.0718 | -0.2506 | 0.6667 |

Results on `qfq.zip`, `limit=10`, `top_k=5`, 15 candidates:

| Model | View | Parse | Match | Valid Sharpe | Keep Rate | Kept Test Sharpe | Dropped Test Sharpe | Rule Agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B | tagged evidence | 1.0 | 1.0 | 1.0 | 0.6000 | 0.0798 | 0.2549 | 0.4667 |

Interpretation:

- Tagged train evidence helps Qwen 3B move beyond the degenerate all-drop policy.
- The `limit=5` tagged 3B run separates kept versus dropped candidates in the right direction on held-out strategy Sharpe.
- The `limit=10` tagged 3B robustness run is less favorable: kept candidates underperform dropped candidates on held-out Sharpe.
- Qwen 1.5B remains too weak for this reasoning task: it ignores tags and continues to mark strong train-strategy candidates as weak.
- Explanation compliance should become an explicit benchmark metric because Qwen sometimes emits non-allowed reason codes such as `moderate_train_strategy` and can keep candidates while citing weak strategy evidence.

## Pipeline Audit After Factor-Pool Expansion

Concern:

- After expanding the factor pool, derived tags or residual labels could make the LLM follow a hidden decision rule instead of performing reasoning.

Audit result:

- Real-market factor rows do not receive `label_keep`; those labels are only attached in the synthetic path.
- Formal `sear llm` uses `build_llm_reasoning_view`, not the diagnostic `build_reasoning_view`.
- The formal LLM view does not expose `label_keep`, `test_ic`, `test_strategy_*`, `decision`, `active_regime`, or `score`.
- Evaluation joins held-out metrics and rule decisions only after the LLM response is parsed.

Issues found:

- The old `build_reasoning_view` is intentionally diagnostic/leaky and includes held-out metrics and rule decisions.
- `sear reason` previously defaulted to this leaky diagnostic view, which was easy to misuse.
- `evidence_tags` are not labels, but they are hand-engineered train-only summaries and can make the LLM follow a tag policy rather than reason over raw structured evidence.

Fix:

- `sear reason` now defaults to the leakage-free numeric LLM view.
- The leaky view must be requested explicitly with `--diagnostic-leaky`.
- `sear llm` now defaults to raw numeric train evidence without `evidence_tags`.
- Tagged evidence must be requested explicitly with `--include-evidence-tags` and should be treated as an ablation.
- `sear llm` audits the LLM reasoning view and raises an error if forbidden keys such as `test_*`, `label_keep`, `decision`, or `score` appear.
- Real-market output now writes `real_market_diagnostic_reasoning_view.json` and `real_market_llm_reasoning_view.json` separately.

Interpretation update:

- The earlier tagged Qwen results should be treated as tag-assisted reasoning, not pure LLM reasoning.
- The clean formal baseline should be rerun with the default numeric-only view.
- A proper benchmark table should report at least two columns: `numeric-only LLM` and `tag-assisted LLM`.

## Formula And Train-Sample LLM View

Input change:

- Each `FactorSpec` now includes a readable `formula` string.
- The clean LLM view now includes `factor_name`, `family`, `formula`, a train-only `train_factor_sample`, and numeric train evidence.
- The default LLM view still excludes `test_*`, `label_keep`, rule `decision`, `score`, and `evidence_tags`.
- `evidence_tags` remain an explicit ablation through `--include-evidence-tags`.

Smoke validation:

- `alpha360` still contains `276` factors.
- Missing formula count is `0`.
- LLM view audit passes with formula and train-factor samples included.

Qwen2.5-3B result on `qfq.zip`, formula + train sample view:

| Model | View | Limit | Top K | Candidates | Parse | Match | Valid Sharpe | Keep Rate | Kept Test Sharpe | Dropped Test Sharpe | Kept Test IC | Dropped Test IC | Rule Agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B | formula + train sample, numeric-only | 5 | 3 | 9 | 1.0 | 1.0 | 1.0 | 0.6667 | 0.2477 | -0.6577 | 0.0282 | -0.0127 | 0.2222 |
| Qwen2.5-3B | formula + train sample, numeric-only | 5 | 5 | 15 | 1.0 | 1.0 | 1.0 | 0.9333 | -0.1121 | -0.5780 | 0.0167 | 0.0003 | 0.2000 |

Notes:

- Running `limit=5`, `top_k=5`, `max_new_tokens=1536` OOMed after formula and train samples increased prompt length.
- Running `limit=5`, `top_k=5`, `factor_sample_size=3`, `max_new_tokens=1024` produced valid-looking JSON but was truncated at candidate `C011`, so it correctly recorded `parse_success=0.0` and `n_decisions=0.0`.
- Increasing the same run to `max_new_tokens=2048` completed all 15 candidates.
- The earlier successful run used `top_k=3`, so it evaluated 9 candidates; the new `top_k=5` run evaluates the full 15-candidate prompt.
- These are clean runs where the LLM sees factor formulas and train-only factor values, without evidence tags.
- Kept candidates outperform dropped candidates on held-out Sharpe in both small clean runs, but the `top_k=5` run is overly permissive: it keeps 14 of 15 candidates, emits `active_regime=uncertain` for every candidate, and reuses the same confidence/rationale pattern.
- The next robustness step is to use a stronger local model or stricter deliberation prompt, then compare numeric-only versus tag-assisted reasoning under the same candidate set.

## Family Leakage Audit

Concern:

- Synthetic labels were assigned by `family`, while the synthetic data-generating process only made a small set of named factors truly active.
- The formal LLM view exposed `family` and `family_summary`, and the prompt encouraged "family-level support", so an LLM could rely on family priors instead of candidate-level evidence.

Fix:

- Synthetic labels are now assigned only by the data-generating factor names: `momentum_20d`, `reversal_5d`, and `volume_surge` are true positives; `range_pct`, `close_to_open`, and `noise_factor` are explicit decoys; all other expanded factors default to inactive for synthetic label accuracy.
- The formal LLM view is now family-blind by default: candidates expose `factor_name`, `formula`, train-only factor samples, and train-only evidence, but not `family`.
- `family_summary` is no longer emitted by default.
- `--include-family` and `--include-family-summary` enable family-assisted ablations only.
- The prompt no longer recommends family-level support in the formal decision rule.

Validation:

- Default LLM view: `has_family=False`, `has_family_summary=False`.
- Ablation view with flags: `has_family=True`, `has_family_summary=True`.
- Example synthetic labels after the fix: `momentum_20d=keep`, `reversal_5d=keep`, `volume_surge=keep`, while `mom_20d`, `volume_ratio_20`, and `noise_factor` are not automatically kept.

## Family-Blind End-to-End Runs

Synthetic benchmark after family-label fix:

- Command: `sear synthetic --output-dir outputs\family_blind_synthetic_default`.
- `n_test_samples=10116`.
- `keep_accuracy=0.9893`, but the learned judge keeps no positive factors, so kept-test metrics are `NaN`.
- Interpretation: after removing family-level labels, plain accuracy is dominated by the many inactive expanded factors and is no longer sufficient.
- Metric fix: synthetic summaries now include keep precision, recall, specificity, balanced accuracy, and confusion-matrix counts for both learned and rule judges.

Real-market benchmark on `qfq.zip`, `limit=10`:

- Command: `sear real --zip-path data\qfq.zip --limit 10 --output-dir outputs\family_blind_real_l10`.
- `n_symbols=10`, `n_factor_rows=2760`.
- `avg_test_ic=-0.0660`.
- `keep_rate=0.1141`.
- `avg_test_strategy_sharpe=0.2973`.
- `avg_test_strategy_sharpe_kept=0.7892`.
- `avg_test_strategy_sharpe_dropped=0.2431`.
- Interpretation: the non-LLM rule baseline still separates higher held-out Sharpe candidates on this real-market slice.

Family-blind Qwen results on `qfq.zip`, `limit=5`:

| Model | Top K | Candidates | Sample | Parse | Match | Keep Rate | Kept Test Sharpe | Dropped Test Sharpe | Kept Test IC | Dropped Test IC | Rule Agreement | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-3B-Instruct | 5 | 15 | 3 | 1.0 | 1.0 | 0.5333 | -0.0746 | -0.2215 | 0.0067 | 0.0259 | 0.4667 | family-blind baseline; still template-like confidence/regime |
| Qwen2.5-7B-Instruct | 5 | 15 | 3 | failed | failed | NaN | NaN | NaN | NaN | NaN | NaN | OOM during generation |
| Qwen2.5-7B-Instruct | 3 | 9 | 1 | 1.0 | 1.0 | 1.0000 | -0.0541 | NaN | 0.0146 | NaN | 0.1111 | smaller run completed but kept all candidates |

Interpretation:

- Removing `family` materially changed Qwen2.5-3B behavior: keep rate fell from `0.9333` in the family-exposed run to `0.5333` in the family-blind run.
- The family-blind 3B run still shows weak positive Sharpe separation, but it does not improve held-out IC separation.
- Explanation quality remains weak: Qwen2.5-3B repeats `confidence=0.65` and `active_regime=uncertain` for every candidate.
- Qwen2.5-7B is not automatically better under the current prompt: full 15-candidate inference OOMs, and the smaller 9-candidate run degenerates to all-keep.

## Agentic Evidence Audit Protocol

Motivation:

- The compact reason-code prompt produced valid JSON but not strong reasoning.
- Qwen2.5-3B repeated `confidence=0.65`, selected `active_regime=uncertain` for every candidate, and reused very similar rationales.
- This means the prior protocol measured JSON compliance and weak decision separation more than agentic reasoning.

Protocol change:

- The formal LLM output now uses `reasoning_protocol=agentic_evidence_audit_v1`.
- Each candidate must provide an `evidence_audit` with:
  - `formula_hypothesis`
  - `support_summary`
  - `counter_evidence`
  - `regime_summary`
  - `decision_logic`
- The prompt explicitly asks the model to check counter-evidence and select non-uncertain regimes when train regime evidence supports it.

New diagnostics:

- `confidence_unique_count`
- `confidence_std`
- `non_uncertain_regime_rate`
- `regime_unique_count`
- `evidence_audit_nonempty_rate`
- `evidence_audit_unique_rate`
- field-level non-empty and uniqueness rates for formula hypothesis, support summary, counter-evidence, regime summary, and decision logic

Interpretation:

- Future agentic-reasoning results should be evaluated on both held-out financial metrics and these explanation diagnostics.
- The older compact-rationale Qwen results remain useful baselines, but should not be presented as strong agentic reasoning.

First bounded agentic-audit run:

- Command: `sear llm --backend hf-local --zip-path data\qfq.zip --limit 5 --top-k 3 --factor-sample-size 2 --model Qwen/Qwen2.5-3B-Instruct --hf-device-map auto --max-new-tokens 2048`.
- Output files: `outputs\qwen25_3b_agentic_bounded_*_l5_t3_s2_2048.*`.
- `parse_success=1.0`, `n_decisions=9`, `match_rate=1.0`.
- `keep_rate=0.3333`.
- `mean_test_ic_kept=0.0333`, `mean_test_ic_dropped=0.0052`.
- `mean_test_strategy_sharpe_kept=0.0036`, `mean_test_strategy_sharpe_dropped=-0.0830`.
- `confidence_unique_count=5`, `confidence_std=0.3029`.
- `non_uncertain_regime_rate=1.0`, `regime_unique_count=3`.
- `evidence_audit_nonempty_rate=1.0`, `evidence_audit_unique_rate=1.0`.
- Field-level audit uniqueness is high for formula/support/counter-evidence, but lower for regime and decision logic.

Interpretation update:

- The bounded agentic-audit protocol fixes the previous template failure: confidence varies, regimes are not all uncertain, and every candidate has a structured audit.
- Held-out selection is directionally better in this small run: kept candidates beat dropped candidates on both test IC and test Sharpe.
- Explanation faithfulness is still imperfect: Qwen sometimes places supportive evidence in `counter_evidence` or negative evidence in `support_summary`.
- The next benchmark component should be an automatic explanation-faithfulness checker that validates whether support/counter/regime statements are consistent with the structured numeric fields.

Explanation-faithfulness checker:

- Added a deterministic scorer for structured audit fields.
- It checks whether `support_summary` is positive-evidence aligned, `counter_evidence` is negative-evidence aligned, predicted regime agrees with train high/low-vol IC contrast, and decision logic does not contradict keep/drop.
- Added `score-response` so existing LLM responses can be rescored without rerunning generation.

Rescored bounded agentic-audit run (`top_k=3`, 9 candidates):

- Response: `outputs\qwen25_3b_agentic_faith_response_l5_t3_s2_2048.json`.
- Rescored summary: `outputs\qwen25_3b_agentic_faith_rescored_summary_l5_t3_s2_2048.json`.
- `faith_support_polarity_ok_rate=0.5556`.
- `faith_counter_polarity_ok_rate=0.4444`.
- `faith_regime_ok_rate=0.5556`.
- `faith_decision_conflict_rate=0.0000`.
- `faithfulness_ok_rate=0.4444`.

Larger bounded agentic-audit run (`top_k=5`, 15 candidates):

- Command: `sear llm --backend hf-local --zip-path data\qfq.zip --limit 5 --top-k 5 --factor-sample-size 1 --model Qwen/Qwen2.5-3B-Instruct --hf-device-map auto --max-new-tokens 4096`.
- Output files: `outputs\qwen25_3b_agentic_faith_*_l5_t5_s1_4096.*`.
- `parse_success=1.0`, `n_decisions=15`, `match_rate=1.0`.
- `keep_rate=0.5333`.
- `mean_test_ic_kept=0.0067`, `mean_test_ic_dropped=0.0259`.
- `mean_test_strategy_sharpe_kept=-0.0746`, `mean_test_strategy_sharpe_dropped=-0.2215`.
- `confidence_unique_count=2`, `confidence_std=0.2993`.
- `non_uncertain_regime_rate=1.0`, `regime_unique_count=3`.
- `evidence_audit_nonempty_rate=1.0`, `evidence_audit_unique_rate=1.0`.
- `faith_support_polarity_ok_rate=1.0000`.
- `faith_counter_polarity_ok_rate=0.9333`.
- `faith_regime_ok_rate=0.5333`.
- `faith_decision_conflict_rate=0.0000`.
- `faithfulness_ok_rate=0.4667`.
- `llm_decision_logic_unique_rate=0.2667`, suggesting residual template behavior in final decision explanations.

Current interpretation:

- The agentic-audit protocol fixes the most obvious template failure: outputs parse, include candidate-specific audits, use multiple regimes, and expose confidence variation.
- On held-out strategy Sharpe, Qwen2.5-3B still separates kept from dropped candidates in the right direction.
- On held-out IC, the larger run is unfavorable: dropped candidates have higher mean test IC than kept candidates.
- Explanation faithfulness remains the main blocker for claiming strong reasoning: overall faithfulness is only about `44%-47%`.
- The next step should compare stronger models under the same family-blind agentic-audit protocol and report both financial selection metrics and faithfulness metrics.
