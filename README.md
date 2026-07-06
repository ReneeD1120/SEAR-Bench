# SEAR-Bench

Structured Evidence Agentic Reasoning for regime-aware factor validity.

## What this repo does

- Reads the provided U.S. equity CSV archive.
- Builds a larger Qlib-inspired alpha pool from daily OHLCV data.
- Expands the pool with AlphaBench-style transformations: rolling means/stds/z-scores, ranks, min-max normalization, lags, spreads, and interactions.
- Summarizes each factor into structured evidence.
- Adds train-only rolling IC evidence so the reasoning layer can judge time-varying validity and factor decay.
- Runs a simple benchmark baseline for factor validity and regime stability.
- Evaluates both prediction evidence and trading evidence: train/test IC, long-short proxy return, Sharpe, cumulative return, and drawdown.
- Provides a synthetic data generator for controlled experiments.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
sear synthetic --output-dir outputs
sear real --zip-path /Users/renee/Downloads/RAFPO/不复权.zip --limit 2 --output-dir outputs
sear reason --zip-path /Users/renee/Downloads/RAFPO/不复权.zip --limit 2 --candidate-count 12
sear llm --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 3 --candidate-count 12 --model Qwen/Qwen3-8B --dry-run
```

For Hugging Face local models:

```bash
pip install -e ".[hf]"
sear llm --backend hf-local --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 3 --candidate-count 12 --model Qwen/Qwen3-8B
```

## Current Design Boundary

The LLM/agent layer is not allowed to generate hidden labels or inspect raw prices directly. It reads the leakage-free view emitted by `sear reason`: factor names, formulas, train-only factor samples, and train-only structured evidence. It then makes a keep/drop and regime explanation. Final judgment still lands on held-out benchmark metrics computed by code.

## Qwen Reasoning

`sear llm` is the first real LLM reasoning entry point. It supports both OpenAI-compatible chat endpoints and Hugging Face local `transformers` models, so Qwen can be called either through a server or directly from the Hugging Face model hub/local checkpoint. The command writes the prompt, calls the model unless `--dry-run` is set, parses strict JSON decisions, and evaluates the model's keep/drop choices against held-out benchmark metrics.

For formal LLM evaluation, `sear llm` uses a leakage-free view: the model sees only factor formulas, train/in-sample factor samples, and train/in-sample structured evidence. Held-out test IC, strategy metrics, hidden labels, and rule decisions are hidden until benchmark scoring. Use `sear reason` to inspect the exact default LLM input view. Use `--include-evidence-tags` only for tag-assisted ablations.

Current default real-data factor bank is `alpha1000`, which contains `1772` readable factor formulas/names. Use `--candidate-count` to control how many factor-symbol candidates are sent to the LLM. For stronger open-source reasoning, prefer a persistent vLLM/OpenAI-compatible Qwen2.5-7B, Qwen3-8B, or larger Qwen server over repeatedly loading Hugging Face models inside one-shot CLI calls.

For a Linux/WSL GPU server with vLLM installed, start Qwen 14B as an OpenAI-compatible endpoint:

```bash
bash scripts/run_qwen14b_vllm.sh
```

Then run SEAR-Bench against it:

```bash
sear llm \
  --backend openai-compatible \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-14B-Instruct \
  --zip-path /Users/renee/Downloads/RAFPO/前复权.zip \
  --limit 3 \
  --candidate-count 12 \
  --factor-sample-size 1 \
  --max-new-tokens 4096
```

On the current Windows lab machine, vLLM tensor parallel is blocked unless WSL/Linux or Docker is installed. Use `TENSOR_PARALLEL_SIZE=2` for Qwen 14B; TP=3 is often incompatible with 14B attention-head partitioning.
