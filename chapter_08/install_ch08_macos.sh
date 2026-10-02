#!/bin/bash

# --- Chapter 8 Environment Setup Script for macOS (Apple Silicon) ---

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch08 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch08 (Python 3.12)"
uv venv .venv_ch08 --python 3.12 --allow-existing
source .venv_ch08/bin/activate

# ── 3. Install macOS-compatible packages ───────────────────────────────────────
# On macOS Apple Silicon:
# - PyTorch with native MPS (Metal) acceleration
# - Evaluation packages: evaluate, rouge_score, bert_score, bleurt, lm-eval
echo "==> Installing packages..."
uv pip install "torch>=2.4.0" "torchvision"
uv pip install "jupyter==1.1.1" "ipykernel==7.2.0" "tqdm" "nltk"
uv pip install "transformers>=4.40.0" "peft>=0.14.0" "datasets" "accelerate"
uv pip install "evaluate==0.4.6" "rouge_score==0.1.2" "bert_score==0.3.13" "lm-eval==0.4.12"
uv pip install git+https://github.com/google-research/bleurt.git

# ── 4. Register the Jupyter kernel ───────────────────────────────────────────
echo "==> Registering Jupyter kernel for ch08..."
python -m ipykernel install --user --name ch08 --display-name ".venv_ch08"

echo "==> macOS Chapter 8 Setup complete!"
