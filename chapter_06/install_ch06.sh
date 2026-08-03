#!/bin/bash

# --- Chapter 6 Environment Setup Script ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch06 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch06 (Python 3.12)"
uv venv .venv_ch06 --python 3.12
source .venv_ch06/bin/activate

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 || uv pip install torch torchvision
uv pip install jupyter ipykernel tqdm 
uv pip install transformers peft trl accelerate bitsandbytes
uv pip install datasets pandas matplotlib scikit-learn
uv pip install unsloth
uv pip install sentence-transformers "torchao>=0.16.0" optimum

python -m ipykernel install --user --name ch06 --display-name ".venv_ch06"

