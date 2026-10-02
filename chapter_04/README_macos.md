# Chapter 04: Fine-Tuning BERT & Encoders on macOS (Apple Silicon MPS)

This guide documents the setup, execution, and architectural adaptations required to run the Chapter 04 data augmentation and encoder notebooks on Apple Silicon Macs (M-series) under macOS.

---

## 1. Quick Setup

Run the automated macOS installation script to create a dedicated virtual environment, install Apple Silicon-optimized packages, and register the Jupyter kernel:

```bash
chmod +x install_ch04_macos.sh
./install_ch04_macos.sh
source .venv_ch04/bin/activate
```

---

## 2. Notebook-by-Notebook macOS Adaptations

| Notebook | Topic | macOS Adaptation |
| :--- | :--- | :--- |
| `listing_4.1-4.10.ipynb` | Data Augmentation & Judging | Runs directly on macOS as-is; `BitsAndBytesConfig` 4-bit NF4 quantization runs natively on Apple Silicon MPS via `device_map="auto"`. |
| `listing_4.11-4.14.ipynb` | Reasoning & Explanation Generation | Runs directly on macOS as-is; `BitsAndBytesConfig` 4-bit NF4 quantization runs natively on Apple Silicon MPS via `device_map="auto"`. |
| `listing_4.15-4.25_macos.ipynb` | Synthetic Data Kit Pipeline | vLLM requires Linux/CUDA. Adapted to connect to local Ollama endpoint (`http://localhost:11434/v1`) using Apple Silicon Metal GPU acceleration. |
