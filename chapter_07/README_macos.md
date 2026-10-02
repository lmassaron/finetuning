# Chapter 07: Preference Tuning (DPO & ORPO) on macOS (Apple Silicon MPS)

This guide documents the setup, execution, and architectural adaptations required to run the Chapter 07 preference tuning notebooks on Apple Silicon Macs (M-series) under macOS.

---

## 1. Quick Setup

Run the automated macOS installation script to create a dedicated virtual environment, install Apple Silicon-optimized packages, and register the Jupyter kernel:

```bash
chmod +x install_ch07_macos.sh
./install_ch07_macos.sh
source .venv_ch07/bin/activate
```

---

## 2. Notebook-by-Notebook macOS Adaptations

| Notebook | Topic | macOS Adaptation |
| :--- | :--- | :--- |
| `listing_7.1-7.6_macos.ipynb` | Direct Preference Optimization (DPO) | Unsloth relies on CUDA/Triton kernels. Adapted using standard Hugging Face `peft` (`get_peft_model`) and `trl.DPOTrainer` with native Apple Silicon Metal acceleration. Hyperparameters, prompts, and evaluation flow are preserved 1-to-1. |
| `listing_7.7-7.18.ipynb` | Odds Ratio Preference Optimization (ORPO) | Runs directly on Apple Silicon MPS as-is (already includes native MPS device selection in original notebook); no separate macOS notebook needed. |
