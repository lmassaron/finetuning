# Chapter 06: Supervised Fine-Tuning (SFT & PEFT) on macOS (Apple Silicon MPS)

This guide documents the setup, execution, and architectural adaptations required to run the Chapter 06 supervised fine-tuning notebooks on Apple Silicon Macs (M-series) under macOS.

---

## 1. Quick Setup

Run the automated macOS installation script to create a dedicated virtual environment, install Apple Silicon-optimized packages, and register the Jupyter kernel:

```bash
chmod +x install_ch06_macos.sh
./install_ch06_macos.sh
source .venv_ch06/bin/activate
```

---

## 2. Notebook-by-Notebook macOS Adaptations

| Notebook | Topic | macOS Adaptation |
| :--- | :--- | :--- |
| `listing_6.1-6.9.ipynb` | SFT with Gemma 2 | Runs directly on macOS as-is; MPS detection, `BitsAndBytesConfig` 4-bit NF4, and `optim="adamw_torch_fused"` run natively on Apple Silicon Metal. |
| `listing_6.10-6.16_macos.ipynb` | PEFT SFT on Multi-Turn QA | Unsloth relies on CUDA/Triton kernels. Adapted using standard Hugging Face `transformers` + `peft` with native Apple Silicon Metal acceleration and `bf16=True`. |
| `listing_6.17-6.24.ipynb` | Text Generation Pipeline | Runs directly on macOS as-is; `BitsAndBytesConfig` 4-bit NF4 and `bf16=True` execute natively on Apple Silicon MPS via `device_map="auto"`. |
| `listing_6.25-6.29_macos.ipynb` | Vision-Language SFT (Qwen2-VL) | Unsloth Vision relies on CUDA/Triton. Adapted using Hugging Face `Qwen2VLForConditionalGeneration` + `peft` with multimodal collator on MPS and `bf16=True`. |
| `listing_6.30-6.34.ipynb` | LoRA Fine-Tuning Gemma 3 1B | Runs directly on macOS as-is; loads in `dtype=torch.bfloat16` with `device_map="auto"` on MPS without requiring a separate macOS notebook. |
