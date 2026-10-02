# Chapter 09: Quantization, Edge Serving, and Inference on macOS (Apple Silicon MPS)

This guide documents the setup, execution, and architectural adaptations required to run the Chapter 09 quantization and inference notebooks on Apple Silicon Macs (M-series) under macOS.

---

## 1. Quick Setup

Run the automated macOS installation script to create a dedicated virtual environment, install Apple Silicon-optimized packages, and register the Jupyter kernel:

```bash
chmod +x install_ch09_macos.sh
./install_ch09_macos.sh
source .venv_ch09/bin/activate
```

---

## 2. Notebook-by-Notebook macOS Adaptations

| Notebook | Topic | macOS Adaptation |
| :--- | :--- | :--- |
| `listing_9.1.ipynb` | Merging LoRA into Base Model | Runs directly on macOS as-is; uses native PyTorch on MPS (`dtype=torch.float16`). |
| `listing_9.2_macos.ipynb` | GPTQ Quantization (GPTQModel) | Uses `device="mps" if torch.backends.mps.is_available() else "cpu"` for Apple Silicon Metal acceleration. |
| `listing_9.2b.ipynb` | GPTQ Quantization (LLM Compressor) | Runs natively on Apple Silicon as-is using `llmcompressor`. |
| `listing_9.3_macos.ipynb` | AWQ Quantization (AutoAWQ) | AutoAWQ requires CUDA C++ kernels and is unavailable on macOS. Directs users to use `listing_9.3b.ipynb`. |
| `listing_9.3b.ipynb` | AWQ Quantization (LLM Compressor) | Runs natively on Apple Silicon as-is using `llmcompressor`. |
| `listing_9.4_macos.ipynb` | Offline Batch Inference | vLLM requires CUDA. Replaced with native Hugging Face Transformers batch generation on MPS (`device="mps"`). |
| `listing_9.5_macos.ipynb` | KV Caching & Offloading | Apple Silicon Unified Memory Architecture (UMA) makes PCIe CPU offloading unnecessary. Uses Transformers on MPS (`device="mps"`). |
| `listing_9.6_macos.ipynb` | Dynamic Multi-LoRA Serving | vLLM LoRARequest replaced with Hugging Face PEFT dynamic adapter switching on MPS (`device="mps"`). |
| `listing_9.7.ipynb` | Local Inference via OpenAI Client | Runs directly on macOS as-is; connects to local OpenAI-compatible endpoint. |
| `listing_9.8.ipynb` | Streaming Inference with Ollama | Runs directly on macOS as-is; uses Ollama with Apple Metal GPU acceleration. |
| `listing_9.9.ipynb` | Ollama via OpenAI API | Runs directly on macOS as-is; interacts with Ollama's `/v1` endpoint. |
