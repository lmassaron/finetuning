#!/bin/bash

# --- Chapter 9 Environment Setup Script ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch09 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch09 (Python 3.12)"
uv venv .venv_ch09 --python 3.12
source .venv_ch09/bin/activate

uv pip install torch torchvision
uv pip install jupyter ipykernel
uv pip install transformers peft
uv pip install vllm openai ollama
uv pip install lmcache --no-deps
uv pip install \
  aiofile aiofiles aiohttp awscrt blake3 \
  fastapi httptools httpx msgspec \
  numba numpy nvtx \
  opentelemetry-api opentelemetry-exporter-otlp \
  opentelemetry-exporter-prometheus opentelemetry-sdk \
  prometheus-client psutil py-cpuinfo pyyaml \
  pyzmq redis safetensors setuptools sortedcontainers \
  transformers uvicorn