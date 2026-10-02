#!/bin/bash

# --- Chapter 9 Environment Setup Script for macOS (Apple Silicon MPS) ---

set -euo pipefail

echo "==> Setting up Chapter 09 virtual environment on macOS..."

# 1. cd to script directory
cd "$(dirname "$0")"

# 2. Ensure uv is available
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# 2. Create the ch09 virtual environment
echo "==> Creating venv: .venv_ch09 (Python 3.12)"
uv venv .venv_ch09 --python 3.12
source .venv_ch09/bin/activate

# 3. Install core dependencies with Apple Silicon MPS support
echo "==> Installing core dependencies..."
uv pip install \
    "torch>=2.4.0" \
    "torchvision" \
    "jupyter==1.1.1" \
    "ipykernel==7.2.0" \
    "transformers>=4.48.0" \
    "peft>=0.14.0" \
    "datasets>=3.0.0" \
    "accelerate>=1.0.0" \
    "openai>=1.60.0" \
    "ollama>=0.4.0" \
    "huggingface_hub" \
    "psutil"

# 4. Install quantization toolkits compatible with macOS
echo "==> Installing quantization libraries (gptqmodel, llmcompressor)..."
uv pip install "gptqmodel" || echo "Note: gptqmodel install skipped or fallback needed"
uv pip install "llmcompressor" || echo "Note: llmcompressor install skipped or fallback needed"

# 5. Register Jupyter kernel
python -m ipykernel install --user --name ch09 --display-name "Python 3 (ch09)"

echo "==> Chapter 09 environment setup complete!"
