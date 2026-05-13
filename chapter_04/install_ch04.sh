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

uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 || uv pip install torch torchvision
uv pip install jupyter ipykernel tqdm Wikipedia-API
uv pip install transformers peft
uv pip install datasets BitsAndBytes
uv pip install vllm synthetic-data-kit