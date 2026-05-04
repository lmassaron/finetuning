# Chapter 9: Model Optimization and Deployment

This directory contains the code listings for Chapter 9, focusing on model quantization, optimization, and serving using various tools and frameworks.

## Notebooks Overview

The following table summarizes the notebooks available in this chapter and the virtual environment required to run each one:

| Notebook | Description | Virtual Environment |
| :--- | :--- | :--- |
| **listing_9.1.ipynb** | Loading a base model and its LoRA adapter using HF Transformers and PEFT. | `.venv_ch09` |
| **listing_9.2.ipynb** | Quantizing a model using GPTQ via the `GPTQModel` library. | `.venv_ch09_autogpt` |
| **listing_9.3.ipynb** | Quantizing a model using AWQ via the `AutoAWQ` library. | `.venv_ch09_autoawq` |
| **listing_9.4.ipynb** | Serving and generating text efficiently using the `vLLM` engine. | `.venv_ch09` |
| **listing_9.5.ipynb** | Implementing KV cache transfer and caching using `vLLM` and `LMCache`. | `.venv_ch09` |
| **listing_9.6.ipynb** | Running multi-LoRA inference using `vLLM`. | `.venv_ch09` |
| **listing_9.7.ipynb** | Interacting with a `vLLM` server using the OpenAI-compatible SDK. | `.venv_ch09` |
| **listing_9.8.ipynb** | Managing and interacting with models using `Ollama`. | `.venv_ch09` |
| **listing_9.9.ipynb** | Using the OpenAI-compatible endpoint provided by `Ollama`. | `.venv_ch09` |

## Environment Setup

To ensure all dependencies are correctly installed and to avoid version conflicts, three separate virtual environments are used. We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable environment management.

### 1. Main Environment (`.venv_ch09`)
Used for Transformers, PEFT, vLLM, LMCache, OpenAI SDK, and Ollama.
```bash
bash install_ch09.sh
```

### 2. GPTQ Environment (`.venv_ch09_autogpt`)
Specialized environment for GPTQ quantization using `gptqmodel`.
```bash
bash install_ch09_autogpt.sh
```

### 3. AutoAWQ Environment (`.venv_ch09_autoawq`)
Specialized environment for AWQ quantization using `autoawq`.
```bash
bash install_ch09_autoawq.sh
```

## Running the Notebooks

1. Activate the appropriate virtual environment:
   ```bash
   source .venv_ch09/bin/activate  # or the specific environment for the notebook
   ```
2. Start the Jupyter server:
   ```bash
   jupyter notebook
   ```
3. In the Jupyter interface, ensure you select the kernel corresponding to the activated virtual environment.
