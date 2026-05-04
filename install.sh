#!/bin/bash
# Corrected Installation script for fine-tuning workshop (aarch64 support restored)

set -e # Exit on error

VENV_NAME=".venv"
PYTHON_VERSION="3.12.13"
export CUDA_HOME=/usr/local/cuda

# 1. Check if uv is installed
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Please install it first."
    exit 1
fi

# Detect OS and GPU
OS_TYPE=$(uname -s)
ARCH=$(uname -m)
HAS_NVIDIA_GPU=false
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    HAS_NVIDIA_GPU=true
fi

echo ">>> Detected OS: $OS_TYPE"
echo ">>> Detected Architecture: $ARCH"
echo ">>> NVIDIA GPU Found: $HAS_NVIDIA_GPU"

echo ">>> Creating virtual environment '$VENV_NAME' with Python $PYTHON_VERSION..."
uv venv "$VENV_NAME" --python "$PYTHON_VERSION" --clear
VENV_PYTHON="$(pwd)/$VENV_NAME/bin/python"

# 2. Install Hardware-Specific ML Libraries
if [ "$OS_TYPE" == "Darwin" ]; then
    echo ">>> Installing macOS-optimized (MPS) stack..."
    uv pip install -U --python "$VENV_PYTHON" \
        "torch" "torchvision" "torchaudio" "transformers" "trl" \
        "peft" "accelerate" "bitsandbytes" "datasets" "evaluate"
    
    if [[ "$ARCH" == 'arm64' ]]; then
        echo ">>> Apple Silicon detected. Installing vLLM base..."
        uv pip install --python "$VENV_PYTHON" "vllm"
        echo "⚠️ Note: vllm-metal requires building from source locally on Macs. Skipping autoawq/exllamav3 as they are CUDA-only."
    fi

elif [ "$HAS_NVIDIA_GPU" = true ]; then
    echo ">>> Installing Linux/CUDA-optimized (NVIDIA GPU) stack..."
    
    PT_CU_VERSION="cu121"
        
    # 1. Install build tools first
    uv pip install --python "$VENV_PYTHON" maturin ninja cmake psutil
    
   # 2. Install Core ML Stack (This forces uv to resolve Torch and vLLM together)
    echo ">>> Installing Core CUDA ML stack..."
    uv pip install -U --python "$VENV_PYTHON" \
        --extra-index-url "https://download.pytorch.org/whl/${PT_CU_VERSION}" \
        --index-strategy unsafe-best-match \
        "torch" "torchvision" "torchaudio" \
        "vllm>=0.6.0" "transformers" "trl" "peft" "accelerate" "bitsandbytes" "datasets" "evaluate"

    # 3. Build Quantization Libraries WITHOUT build isolation
    # This allows their poorly written setup.py files to "see" the Torch we just installed.
    echo ">>> Building Quantization Libraries from source..."
    export MAX_JOBS=4
    export NINJA_BUILD_MAX_JOBS=4
    uv pip install -U --python "$VENV_PYTHON" --no-build-isolation \
        --extra-index-url "https://download.pytorch.org/whl/${PT_CU_VERSION}" \
        --index-strategy unsafe-best-match \
        "autoawq" "exllamav3" "gptqmodel" "unsloth" "xformers"

    # Install Flash Attention without build isolation to prevent system lock-ups
    #echo ">>> Installing Flash Attention..."
    #export MAX_JOBS=2
    #export NINJA_BUILD_MAX_JOBS=2
    #uv pip install -U --python "$VENV_PYTHON" flash-attn --no-build-isolation

else
    echo ">>> Installing Linux CPU/VPU stack..."
    echo "⚠️ Note: For Intel VPU support, you will manually need to install 'optimum-intel' and 'openvino'."
    uv pip install -U --python "$VENV_PYTHON" \
        "transformers" "torch" "torchvision" "trl" "peft" "accelerate" "datasets" "evaluate"
fi

# 3. Install Data Science Libraries (Common)
echo ">>> Installing common data science libraries..."
uv pip install -U --python "$VENV_PYTHON" \
    "pandas" "numpy" "scikit-learn" "matplotlib" "tqdm" "tenacity" \
    "sentencepiece" "sentence-transformers" "optimum" "huggingface_hub" "torchao"

# 4. Install Jupyter
echo ">>> Installing Jupyter..."
uv pip install -U --python "$VENV_PYTHON" "jupyter" "ipykernel"
"$VENV_PYTHON" -m ipykernel install --user --name "fine-tuning-workshop" --display-name "Python (Fine-Tuning)"

echo "✅ Environment setup complete!"