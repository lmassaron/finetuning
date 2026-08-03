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

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 || uv pip install torch torchvision
uv pip install jupyter ipykernel tqdm unsloth torchao "optimum<2.0.0"
uv pip install "transformers<5.0.0" "peft==0.14.0" accelerate "trl<0.12.0" liger-kernel
uv pip install datasets BitsAndBytes pandas
BUILD_CUDA_EXT=0 uv pip install auto-gptq --no-build-isolation
uv pip uninstall -y gptqmodel || true


echo "==> Registering Jupyter kernel for ch05..."
python -m ipykernel install --user --name ch05 --display-name ".venv_ch05"