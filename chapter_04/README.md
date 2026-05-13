# Chapter 4: Data Augmentation and Synthetic Data Generation

This directory contains the code listings for Chapter 4, focusing on building high-quality datasets through data augmentation and synthetic data generation. It covers techniques for fetching medical data, augmenting financial datasets, and adding reasoning to existing data using LLMs.

## Notebooks Overview

The following table summarizes the notebooks available in this chapter and the virtual environment required to run each one:

| Notebook | Description | Virtual Environment |
| :--- | :--- | :--- |
| **building_medical_dataset.ipynb** | Building a medical dataset by fetching content from Wikipedia and generating QA pairs using vLLM. | `.venv_ch04` |
| **augmenting.ipynb** | Augmenting financial news headlines by rephrasing, replacing PII, and diversifying numerical figures. | `.venv_ch04` |
| **adding_reasoning.ipynb** | Enhancing a financial dataset by adding reasoning and explanations using a teacher model. | `.venv_ch04` |

## Environment Setup

To ensure all dependencies are correctly installed and to avoid version conflicts, a dedicated virtual environment is used. We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable environment management.

### Main Environment (`.venv_ch04`)
Used for all data augmentation and synthetic data generation tasks in this chapter.
```bash
bash install_ch04.sh
```

## Running the Notebooks

1. Activate the appropriate virtual environment:
   ```bash
   source .venv_ch04/bin/activate
   ```
2. Start the Jupyter server:
   ```bash
   jupyter notebook
   ```
3. In the Jupyter interface, ensure you select the kernel corresponding to the activated virtual environment.
