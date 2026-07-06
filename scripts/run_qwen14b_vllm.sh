#!/usr/bin/env bash
set -euo pipefail

# Run from a Linux/WSL machine with vLLM installed.
# Qwen 14B attention heads are more compatible with TP=2 than TP=3.
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code
