#!/bin/bash

# --- Chapter 10 Environment Setup Script for macOS (Apple Silicon MPS) ---

set -euo pipefail

# 1. cd to script directory
cd "$(dirname "$0")"

echo "==> Setting up Chapter 10 virtual environment on macOS..."

# 2. Ensure uv is available
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# 3. Create the ch10 virtual environment
echo "==> Creating venv: .venv_ch10 (Python 3.12)"
uv venv .venv_ch10 --python 3.12
source .venv_ch10/bin/activate

# 4. Install dependencies with Apple Silicon MPS support
echo "==> Installing core dependencies..."
uv pip install \
    "torch>=2.4.0" \
    "torchvision" \
    "jupyter==1.1.1" \
    "ipykernel==7.3.0" \
    "tqdm>=4.66.0" \
    "transformers>=4.48.0" \
    "peft>=0.14.0" \
    "trl>=0.14.0" \
    "accelerate>=1.0.0" \
    "datasets>=3.0.0"

# 5. Register Jupyter kernel
echo "==> Registering Jupyter kernel for ch10..."
python -m ipykernel install --user --name ch10 --display-name "Python 3 (ch10)"

echo "==> Chapter 10 environment setup complete!"
