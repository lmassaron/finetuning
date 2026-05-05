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
echo "==> Creating venv: ch09 llm-compressor (Python 3.12)"
uv venv .venv_ch09_llm_compressor --python 3.12
source .venv_ch09_llm_compressor/bin/activate

# ── 3. Install packages ───────────────────────────────────────────────────────
echo "==> Installing torch and dependencies..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu130 || uv pip install torch

uv pip install jupyter ipykernel
uv pip install llmcompressor transformers accelerate datasets protobuf

uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
# ── 4. Register the Jupyter kernel ───────────────────────────────────────────
#echo "==> Registering Jupyter kernel..."
#python -m ipykernel install --user --name ch09_llm_compressor --display-name ".venv_ch09_llm_compressor"

echo "==> Setup complete!"