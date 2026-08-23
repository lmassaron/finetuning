# Chapter 7: Preference Alignment and Reinforcement Learning

This directory contains the code listings for Chapter 7, focusing on preference alignment (DPO) and reinforcement learning (GRPO) techniques for fine-tuning Large Language Models.

## Notebooks Overview

The following table summarizes the notebooks available in this chapter:

| Notebook | Description | Virtual Environment |
| :--- | :--- | :--- |
| **listing_7.1-7.6.ipynb** | Direct Preference Optimization (DPO) using Unsloth and TRL on preference pair datasets. | `.venv_ch07` |
| **listing_7.7-7.18.ipynb** | Group Relative Policy Optimization (GRPO) for reasoning alignment on GSM8K math problems. | `.venv_ch07` |

## Environment Setup

To ensure all dependencies are correctly installed and to avoid version conflicts, a dedicated virtual environment is provided. We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable environment management.

### Virtual Environment (`.venv_ch07`)
Used for all preference alignment and RL fine-tuning tasks in this chapter.
```bash
bash install_ch07.sh
```

## Running the Notebooks

1. Activate the virtual environment:
   ```bash
   source .venv_ch07/bin/activate
   ```
2. Start Jupyter notebook or select kernel `.venv_ch07` in your IDE:
   ```bash
   jupyter notebook
   ```
