#!/bin/bash

# --- Chapter 10 Environment Setup Script ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch10 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch10 (Python 3.12)"
uv venv .venv_ch10 --python 3.12
source .venv_ch10/bin/activate

uv pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130 || uv pip install "torch==2.11.0" "torchvision==0.26.0"
uv pip install "jupyter==1.1.1" "ipykernel==7.3.0" "tqdm==4.70.0"
uv pip install "transformers==4.57.6" "peft==0.20.0" "trl==0.24.0" "accelerate==1.14.0" "bitsandbytes==0.50.0"
uv pip install "datasets==4.3.0"

echo "==> Registering Jupyter kernel for ch10..."
python -m ipykernel install --user --name ch10 --display-name ".venv_ch10"
