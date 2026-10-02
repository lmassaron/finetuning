# Chapter 08: Evaluation and LLM-as-a-Judge on macOS (Apple Silicon MPS)

This guide documents the setup, execution, and architectural adaptations required to run the Chapter 08 evaluation notebooks on Apple Silicon Macs (M-series) under macOS.

---

## 1. Quick Setup

Run the automated macOS installation script to create a dedicated virtual environment, install Apple Silicon-optimized packages, and register the Jupyter kernel:

```bash
chmod +x install_ch08_macos.sh
./install_ch08_macos.sh
source .venv_ch08/bin/activate
```

---

## 2. Notebook-by-Notebook macOS Adaptations

| Notebook | Topic | macOS Adaptation |
| :--- | :--- | :--- |
| `listing_8.1.ipynb` | BLEU, ROUGE, and METEOR | Pure CPU evaluation; runs directly on macOS as-is. |
| `listing_8.2.ipynb` | BERTScore and BLEURT | Hugging Face Evaluate + ARM64 TensorFlow; runs directly on macOS as-is. |
| `listing_8.3-8.7.ipynb` | LLM-as-a-Judge Pipeline | Automatic device routing (`device_map="auto"`) places models onto `mps:0` on Apple Silicon. Runs directly on macOS as-is. |
| `listing_8.8_macos.ipynb` | GSM8K with EleutherAI LM-Eval | Specifies `device="mps"` with `dtype="bfloat16"` for Apple Silicon Metal acceleration (since `device="auto"` in `lm-eval` falls back to CPU on macOS). |
