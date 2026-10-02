#!/bin/bash

# --- Chapter 5 Environment Setup Script for macOS (Apple Silicon) ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch05 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch05 (Python 3.12)"
uv venv .venv_ch05 --python 3.12 --allow-existing
source .venv_ch05/bin/activate

# ── 3. Install macOS-compatible packages ───────────────────────────────────────
# On macOS Apple Silicon:
# - PyTorch runs natively on Apple Silicon with MPS (Metal)
# - Unsloth, Liger-Kernel, and Auto-GPTQ require Linux/CUDA Triton kernels and are omitted
# - Hugging Face PEFT, TRL, Accelerate, and Transformers handle all fine-tuning on MPS
echo "==> Installing PyTorch and dependencies..."
uv pip install "torch>=2.4.0" "torchvision"
uv pip install "jupyter==1.1.1" "ipykernel==7.3.0" "tqdm==4.70.0" "psutil"
uv pip install "transformers==4.57.6" "peft>=0.14.0" "accelerate>=1.0.0" "trl>=0.11.4"
uv pip install "datasets==4.3.0" "pandas>=2.2.0"

# ── 4. Register the Jupyter kernel ───────────────────────────────────────────
echo "==> Registering Jupyter kernel for ch05..."
python -m ipykernel install --user --name ch05 --display-name ".venv_ch05"

echo "==> macOS Chapter 5 Setup complete!"
