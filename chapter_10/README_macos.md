# Chapter 10: Multi-Agent Systems on macOS (Apple Silicon MPS)

This guide documents the setup, execution, and architectural adaptations required to run the Chapter 10 multi-agent fine-tuning pipeline and dynamic agent CLI on Apple Silicon Macs (M-series) under macOS.

---

## 1. Quick Setup

Run the automated macOS installation script to create a dedicated virtual environment, install Apple Silicon-optimized packages, and register the Jupyter/CLI kernel:

```bash
chmod +x install_ch10_macos.sh
./install_ch10_macos.sh
source .venv_ch10/bin/activate
```

---

## 2. Technical Divergences & macOS Adaptations

### A. Precision & bfloat16 Execution
- **Linux/CUDA**: The original scripts use `torch.cuda.is_bf16_supported()`, which evaluates to `False` on macOS where CUDA is absent.
- **macOS Adaptation**: In `train_loras_macos.py` and `coding_agent_macos.py`, precision is configured with `bf16=True` and `compute_dtype = torch.bfloat16 if torch.backends.mps.is_available() else torch.float16`, ensuring full BrainFloat support on Apple Silicon MPS.

### B. Device Routing & Memory Clearance
- **Linux/CUDA**: In `coding_agent.py:98`, `inputs = self.tokenizer(text=text, return_tensors="pt").to("cuda")` was hardcoded to `"cuda"`, and cache clearance invoked `torch.cuda.empty_cache()`.
- **macOS Adaptation**: In `coding_agent_macos.py`, inputs are routed dynamically via `.to(self.model.device)`, and memory cache is freed via `torch.mps.empty_cache()`. Both `BitsAndBytesConfig` 4-bit NF4 quantization and `optim="paged_adamw_8bit"` are preserved.

---

## 3. Running the Multi-Agent System

```bash
# Run the multi-agent CLI (uses Qwen/Qwen3-4B with planner, coder, and reviewer LoRA adapters):
python coding_agent_macos.py "Write a Python function to solve the classic 'FizzBuzz' problem in fizzbuzz.py"

# Or run the LoRA training pipeline:
python train_loras_macos.py
```
