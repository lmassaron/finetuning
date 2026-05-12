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

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 || uv pip install torch torchvision
uv pip install jupyter ipykernel
uv pip install transformers peft
uv pip install \
    evaluate \
    rouge_score \
    bert_score \
    git+https://github.com/google-research/bleurt.git \
    lm-eval