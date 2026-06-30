# SEAR-Bench

Structured Evidence Agentic Reasoning for regime-aware factor validity.

## What this repo does

- Reads the provided U.S. equity CSV archive.
- Builds candidate alpha factors from daily OHLCV data.
- Summarizes each factor into structured evidence.
- Runs a simple benchmark baseline for factor validity and regime stability.
- Provides a synthetic data generator for controlled experiments.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m seae.cli summarize --zip-path /Users/renee/Downloads/RAFPO/不复权.zip
```
