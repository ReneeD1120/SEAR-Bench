# SEAR-Bench

Structured Evidence Agentic Reasoning for regime-aware factor validity.

## What this repo does

- Reads the provided U.S. equity CSV archive.
- Builds a larger Qlib-inspired alpha pool from daily OHLCV data.
- Expands the pool with AlphaBench-style transformations: rolling z-scores, ranks, min-max normalization, lags, spreads, and interactions.
- Summarizes each factor into structured evidence.
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
sear reason --zip-path /Users/renee/Downloads/RAFPO/不复权.zip --limit 2 --top-k 5
sear llm --zip-path /Users/renee/Downloads/RAFPO/前复权.zip --limit 3 --top-k 3 --model Qwen/Qwen3-8B --dry-run
```

## Current Design Boundary

The LLM/agent layer is not allowed to generate hidden labels or inspect raw prices directly. It should only read the structured evidence view emitted by `sear reason`, then make a keep/drop and regime explanation. Final judgment still lands on benchmark metrics computed by code.

## Qwen Reasoning

`sear llm` is the first real LLM reasoning entry point. It supports any OpenAI-compatible chat endpoint, so Qwen can be served locally by vLLM, llama.cpp server, or another compatible runtime. The command writes the prompt, calls the model unless `--dry-run` is set, parses strict JSON decisions, and evaluates the model's keep/drop choices against held-out benchmark metrics.
