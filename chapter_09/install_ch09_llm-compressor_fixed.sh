#!/bin/bash

# --- Chapter 9 Environment Setup Script (GPU Fixed) ---
# This script ensures that Torch is installed with CUDA support on DGX Spark
# and that llm-compressor is patched to handle Blackwell GPU metric limitations.

set -euo pipefail

# ── 1. Ensure uv is available ──────────────────────────────────────────────────
echo "==> Checking uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

# ── 2. Create the ch09 virtual environment ────────────────────────────────────
echo "==> Creating venv: ch09 llm-compressor (Python 3.12)"
rm -rf .venv_ch09_llm_compressor
uv venv .venv_ch09_llm_compressor --python 3.12
source .venv_ch09_llm_compressor/bin/activate

# ── 3. Install packages ───────────────────────────────────────────────────────
echo "==> Installing packages with GPU support..."
# Step 3a: Install llmcompressor and its dependencies. 
# uv will initially pull a CPU version of torch due to llmcompressor constraints.
uv pip install \
    "llmcompressor==0.10.0.2" \
    transformers \
    accelerate \
    datasets \
    protobuf \
    jupyter \
    ipykernel

# Step 3b: Force-reinstall CUDA-enabled Torch.
# This bypasses llmcompressor's strict <=2.10.0 requirement, which is safe for these notebooks.
echo "==> Forcing CUDA Torch installation (cu130)..."
uv pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.11.0+cu130" \
    "torchvision==0.26.0+cu130" \
    "torchaudio==2.11.0+cu130"

# ── 4. Patch llmcompressor to handle NVML errors on DGX Spark ────────────────
echo "==> Patching llmcompressor for Blackwell GPU support..."
PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
METRIC_LOGGING_PY="$PYTHON_SITE_PACKAGES/llmcompressor/utils/metric_logging.py"

if [ -f "$METRIC_LOGGING_PY" ]; then
    python - <<EOF
import os
path = "$METRIC_LOGGING_PY"
with open(path, "r") as f:
    content = f.read()

# Target the block that calls pynvml.nvmlDeviceGetMemoryInfo(handle)
# and wrap it in a try-except to handle NVMLError_NotSupported on DGX Spark.
old_block = """                handle = pynvml.nvmlDeviceGetHandleByIndex(id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_usage_percentage = mem_info.used / mem_info.total
                total_memory_gb = mem_info.total / (1e9)
                usage.append(GPUMemory(id, memory_usage_percentage, total_memory_gb))"""

new_block = """                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_usage_percentage = mem_info.used / mem_info.total
                    total_memory_gb = mem_info.total / (1e9)
                    usage.append(GPUMemory(id, memory_usage_percentage, total_memory_gb))
                except Exception:
                    # Fallback for systems where NVML memory info is unsupported (e.g. DGX Spark / GB10)
                    usage.append(GPUMemory(id, 0.0, 0.0))"""

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(path, "w") as f:
        f.write(content)
    print("Successfully patched llmcompressor/utils/metric_logging.py")
else:
    print("Target block not found. Checking if already patched...")
    if "except Exception:" in content:
        print("Patch already present.")
    else:
        print("Warning: Could not apply patch. NVML errors might still occur.")
EOF
else
    echo "Warning: $METRIC_LOGGING_PY not found. Skipping patch."
fi

# ── 5. Register the Jupyter kernel ───────────────────────────────────────────
echo "==> Registering Jupyter kernel..."
python -m ipykernel install --user --name ch09_llm_compressor --display-name ".venv_ch09_llm_compressor"

echo "==> Setup complete! Environment is ready for GPU quantization."
