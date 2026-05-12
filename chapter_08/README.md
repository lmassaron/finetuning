# Chapter 8: Evaluation and Benchmarking

This directory contains the code listings for Chapter 8, focusing on evaluating and benchmarking fine-tuned models. It covers classical NLP metrics, modern embedding-based metrics, and the use of LLMs as judges, concluding with standard benchmarking harnesses.

## Notebooks Overview

The following table summarizes the notebooks available in this chapter and the virtual environment required to run each one:

| Notebook | Description | Virtual Environment |
| :--- | :--- | :--- |
| **listing_8.1.ipynb** | Classical NLP Evaluation Metrics BLEU, ROUGE, and METEOR. | `.venv_ch08` |
| **listing_8.2.ipynb** | Evaluating the sensitivity of BERTScore and BLEURT to adversarial example pairs. | `.venv_ch08` |
| **listing_8.3-8.7.ipynb** | Setting up a benchmark, crafting generation functions, and using an LLM as a judge (Listing 8.3-8.7). | `.venv_ch08` |
| **listing_8.8.ipynb** | Running the GSM8K benchmark using the EleutherAI LM-Eval Harness. | `.venv_ch08` |

## Environment Setup

To ensure all dependencies are correctly installed and to avoid version conflicts, a dedicated virtual environment is used. We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable environment management.

### Main Environment (`.venv_ch08`)
Used for all evaluation metrics and benchmarking tools in this chapter.
```bash
bash install_ch08.sh
```

## Running the Notebooks

1. Activate the appropriate virtual environment:
   ```bash
   source .venv_ch08/bin/activate
   ```
2. Start the Jupyter server:
   ```bash
   jupyter notebook
   ```
3. In the Jupyter interface, ensure you select the kernel corresponding to the activated virtual environment.
