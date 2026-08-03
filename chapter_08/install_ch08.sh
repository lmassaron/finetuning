#!/bin/bash

# --- Chapter 8 Environment Setup Script ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch08 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch08 (Python 3.12)"
uv venv .venv_ch08 --python 3.12
source .venv_ch08/bin/activate

uv pip install "torch==2.11.0+cu130" "torchvision==0.26.0+cu130" --index-url https://download.pytorch.org/whl/cu130 || uv pip install "torch==2.11.0+cu130" "torchvision==0.26.0+cu130"
uv pip install "jupyter==1.1.1" "ipykernel==7.2.0"
uv pip install "transformers==5.8.0" "peft==0.19.1"
uv pip install \
    "evaluate==0.4.6" \
    "rouge_score==0.1.2" \
    "bert_score==0.3.13" \
    git+https://github.com/google-research/bleurt.git \
    "lm-eval==0.4.12"