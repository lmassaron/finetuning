# Chapter 05: Pre-training & Domain Adaptation on macOS (Apple Silicon MPS)

This guide documents the setup, execution, and architectural adaptations required to run the Chapter 05 pre-training and domain adaptation notebooks on Apple Silicon Macs (M-series) under macOS.

---

## 1. Quick Setup

Run the automated macOS installation script to create a dedicated virtual environment, install Apple Silicon-optimized packages, and register the Jupyter kernel:

```bash
chmod +x install_ch05_macos.sh
./install_ch05_macos.sh
source .venv_ch05/bin/activate
```

---

## 2. Notebook-by-Notebook macOS Adaptations

| Notebook | Topic | macOS Adaptation |
| :--- | :--- | :--- |
| `listing_5.1-5.5_macos.ipynb` | PEFT & SFT Foundations | `check_used_memory()` reports MPS allocated memory via `torch.mps`. Preserves `BitsAndBytesConfig` 4-bit NF4 quantization on Apple Silicon MPS. |
| `listing_5.6_macos.ipynb` | LoRA Fine-Tuning | Unsloth relies on CUDA/Triton kernels. Adapted using standard Hugging Face `transformers` + `peft` with native Apple Silicon Metal acceleration. |
| `listing_5.7_macos.ipynb` | Quantized Model Fine-Tuning | Fine-tuning through 4-bit GPTQ weights requires CUDA kernels for backpropagation. On Apple Silicon, we fine-tune the model in `bfloat16` directly using Unified Memory. |
