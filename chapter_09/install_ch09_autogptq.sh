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
echo "==> Creating venv: ch09 AutoGPT (Python 3.12)"
uv venv .venv_ch09_autogpt --python 3.12
source .venv_ch09_autogpt/bin/activate

uv pip install torch torchvision
uv pip install jupyter ipykernel
uv pip install gptqmodel transformers