#!/bin/bash

# --- Chapter 4 Environment Setup Script ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch04 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch04 (Python 3.12)"
uv venv .venv_ch04 --python 3.12
source .venv_ch04/bin/activate

uv pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130 || uv pip install "torch==2.11.0" "torchvision==0.26.0"
uv pip install "jupyter==1.1.1" "ipykernel==7.2.0" "tqdm==4.67.3" "Wikipedia-API==0.14.1"
uv pip install "transformers==5.8.1" "peft==0.19.1"
uv pip install "datasets==4.8.5" "BitsAndBytes==0.49.2"
uv pip install "vllm==0.20.2" "synthetic-data-kit==0.0.5"