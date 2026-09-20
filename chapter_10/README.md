# Chapter 10: Autonomous Agents and Multi-Adapter Fine-Tuning

This directory contains the code listings and scripts for Chapter 10, focusing on building autonomous coding agents using specialized, fine-tuned LoRA adapters. It demonstrates how to train multiple role-specific adapters (Planner, Coder, and Reviewer) on a single quantized base model and dynamically swap them at runtime to accomplish multi-step software engineering tasks.

## Code Overview

The following table summarizes the scripts and modules available in this chapter and the virtual environment required to run each one:

| Script / Module | Description | Virtual Environment |
| :--- | :--- | :--- |
| **`tools.py`** | Environment tool definitions (`read_file`, `write_file`, `list_files`) and schema registry for agent tool use. | `.venv_ch10` |
| **`train_loras.py`** | Training pipeline for three specialized LoRA adapters (Planner, Coder, Reviewer) on `Qwen/Qwen3-4B` using 4-bit QLoRA and TRL's `SFTTrainer`. | `.venv_ch10` |
| **`coding_agent.py`** | Multi-adapter autonomous coding agent CLI that dynamically swaps PEFT LoRA adapters across planning, code generation, tool execution, and QA review loops. | `.venv_ch10` |

## Environment Setup

To ensure all dependencies are correctly installed and to avoid version conflicts, a dedicated virtual environment is used. We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable environment management.

### Main Environment (`.venv_ch10`)
Used for training LoRA adapters, managing dynamic PEFT adapter switching, and running the autonomous agent.
```bash
bash install_ch10.sh
```

## Running the Scripts

1. **Activate the virtual environment:**
   ```bash
   source .venv_ch10/bin/activate
   ```

2. **(Optional) Train the specialized LoRA adapters:**
   Run the training pipeline to fine-tune the three LoRA adapters locally using TRL:
   ```bash
   python train_loras.py
   ```
   This script fine-tunes:
   - **Planner:** Specialized on `Open-Orca/OpenOrca` for task decomposition and step planning.
   - **Coder:** Specialized on `NousResearch/hermes-function-calling-v1` for structured JSON tool invocation.
   - **Reviewer:** Specialized on `m-a-p/CodeFeedback-Filtered-Instruction` for automated code critique and QA evaluation.

3. **Run the Autonomous Coding Agent:**
   Execute the agent on the default FizzBuzz generation and testing task:
   ```bash
   python coding_agent.py
   ```
   Or pass a custom natural language instruction:
   ```bash
   python coding_agent.py "Write a Python function to compute the Fibonacci sequence, save it to fibonacci.py, and generate test cases in tests/test_fibonacci.py"
   ```

4. **Interactive Development (Optional):**
   To experiment in Jupyter notebooks or an interactive shell using the chapter environment:
   ```bash
   jupyter notebook
   ```
   In the Jupyter interface, select the `.venv_ch10` kernel.
