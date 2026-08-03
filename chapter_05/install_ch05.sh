#!/bin/bash

# --- Chapter 5 Environment Setup Script ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch05 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch05 (Python 3.12)"
uv venv .venv_ch05 --python 3.12
source .venv_ch05/bin/activate

uv pip install "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cu130 || uv pip install "torch==2.11.0" "torchvision==0.26.0"
uv pip install "jupyter==1.1.1" "ipykernel==7.3.0" "tqdm==4.70.0" "unsloth==2026.8.1" "torchao==0.17.0" "optimum==1.27.0"
uv pip install "transformers==4.57.6" "peft==0.14.0" "accelerate==1.14.0" "trl==0.11.4" "liger-kernel==0.8.1"
uv pip install "datasets==4.3.0" "BitsAndBytes==0.50.0" "pandas==3.0.5"
BUILD_CUDA_EXT=0 uv pip install "auto-gptq==0.7.1" --no-build-isolation
uv pip uninstall -y gptqmodel || true


echo "==> Registering Jupyter kernel for ch05..."
python -m ipykernel install --user --name ch05 --display-name ".venv_ch05"