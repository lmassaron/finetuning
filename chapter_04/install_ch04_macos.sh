#!/bin/bash

# --- Chapter 4 Environment Setup Script for macOS (Apple Silicon) ---

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

# ── 3. Install macOS-compatible packages ───────────────────────────────────────
# On macOS Apple Silicon:
# - Install PyTorch with native MPS (Metal Performance Shaders) support
# - vLLM is excluded because it requires Linux/CUDA (nvidia-cudnn-frontend)
# - synthetic-data-kit is installed and pairs with Ollama for local LLM serving
echo "==> Installing PyTorch and dependencies..."
uv pip install "torch>=2.4.0" "torchvision"
uv pip install "jupyter==1.1.1" "ipykernel==7.2.0" "tqdm==4.67.3" "Wikipedia-API==0.14.1"
uv pip install "transformers==5.8.1" "peft==0.19.1" "bitsandbytes>=0.44.0"
uv pip install "datasets==4.8.5" "pandas>=2.2.0"
uv pip install "synthetic-data-kit==0.0.5"

# ── 4. Register the Jupyter kernel ───────────────────────────────────────────
echo "==> Registering Jupyter kernel for ch04..."
python -m ipykernel install --user --name ch04 --display-name ".venv_ch04"

echo "==> macOS Setup complete!"
