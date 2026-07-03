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
