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

uv pip install "torch==2.11.0+cu130" "torchvision==0.26.0+cu130" --index-url https://download.pytorch.org/whl/cu130 || uv pip install "torch==2.11.0+cu130" "torchvision==0.26.0+cu130"
uv pip install "jupyter==1.1.1" "ipykernel==7.2.0"
uv pip install "transformers==5.7.0" "peft==0.19.1"
uv pip install "vllm==0.20.1" "openai==2.34.0" "ollama==0.6.2"
uv pip install "lmcache==0.4.4" --no-deps
uv pip install \
  "aiofile==3.9.0" "aiofiles==25.1.0" "aiohttp==3.13.5" "awscrt==0.32.2" "blake3==1.0.8" \
  "fastapi==0.136.1" "httptools==0.7.1" "httpx==0.28.1" "msgspec==0.21.1" \
  "numba==0.65.0" "numpy==2.3.5" "nvtx==0.2.15" \
  "opentelemetry-api==1.41.1" "opentelemetry-exporter-otlp==1.41.1" \
  "opentelemetry-exporter-prometheus==0.62b1" "opentelemetry-sdk==1.41.1" \
  "prometheus-client==0.25.0" "psutil==7.2.2" "py-cpuinfo==9.0.0" "pyyaml==6.0.3" \
  "pyzmq==27.1.0" "redis==7.4.0" "safetensors==0.7.0" "setuptools==80.10.2" "sortedcontainers==2.4.0" \
  "transformers==5.7.0" "uvicorn==0.46.0"