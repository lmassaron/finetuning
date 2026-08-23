#!/bin/bash

# --- Chapter 7 Environment Setup Script ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch07 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch07 (Python 3.12)"
uv venv .venv_ch07 --python 3.12
source .venv_ch07/bin/activate

uv pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130 || uv pip install "torch==2.11.0" "torchvision==0.26.0"
uv pip install "jupyter==1.1.1" "ipykernel==7.3.0" "tqdm==4.70.0"
uv pip install "transformers==5.5.0" "peft==0.20.0" "trl==0.24.0" "accelerate==1.14.0" "bitsandbytes==0.50.0"
uv pip install "datasets==4.3.0" "pandas==3.0.5" "matplotlib==3.11.1"
uv pip install "unsloth==2026.8.1"
uv pip install "vllm==0.20.2"

echo "==> Registering Jupyter kernel for ch07..."
python -m ipykernel install --user --name ch07 --display-name ".venv_ch07"
