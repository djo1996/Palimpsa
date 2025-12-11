<div align="center">
<img width="600" alt="Palimpsa Logo" src="https://github.com/user-attachments/assets/7fa41f32-0976-42c9-8d32-2a602e56289f" />

# Palimpsa
### Learning to Remember, Learn, and Forget in Attention-Based Models

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-blue)](https://arxiv.org/abs/2504.13569)
[![Framework](https://img.shields.io/badge/Built%20On-Flame%20%26%20FLA-firebrick)](https://github.com/fla-org/flame)
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

**Palimpsa** is a novel attention mechanism that views In-Context Learning (ICL) as a continual learning problem. It introduces **Bayesian Metaplasticity** to transformer architectures—dynamically adjusting the plasticity of memory states based on their uncertainty.

This approach solves the **stability-plasticity dilemma** in linear attention:
- **Remembering:** Preserves critical past information using importance-weighted updates.
- **Forgetting:** Prevents "catastrophic remembering" (loss of plasticity) by gradually releasing stale information via a Bayesian forgetting mechanism.

Built as a plugin for [🔥 Flame](https://github.com/fla-org/flame), Palimpsa features custom **Triton kernels** that implement a chunk-wise parallel scan, achieving training throughput comparable to Mamba2 while offering superior performance on commonsense reasoning tasks at small scales (170M/340M).

---

## 🛠️ Setup: The "Palimpsa_Lab" Workspace

To ensure full compatibility with the bleeding-edge versions of `fla` and `flame` required for this research, we recommend setting up a dedicated workspace named `Palimpsa_Lab`.

This setup installs all libraries in **editable mode**, allowing you to inspect or modify the core training engine and kernels alongside your Palimpsa experiments.

### 1. Create Workspace & Environment
First, create the lab directory and a fresh virtual environment.

```bash
# 1. Create the working directory
mkdir Palimpsa_Lab
cd Palimpsa_Lab

# 2. Create and activate a virtual environment
python -m venv palimpsa_env
source palimpsa_env/bin/activate  # On Windows use: palimpsa_env\Scripts\activate

# 3. Upgrade pip and install build tools (Critical for Triton kernels)
pip install --upgrade pip
pip install numpy packaging ninja
```

### 2. Install Research Stack
Install dependencies in this exact order to prevent version conflicts.

```bash
# [CRITICAL] Install the specific TorchTitan commit required by Flame
pip install git+[https://github.com/pytorch/torchtitan.git@0b44d4c](https://github.com/pytorch/torchtitan.git@0b44d4c)

# 1. Flash Linear Attention (FLA) - The modeling backend
git clone [https://github.com/fla-org/flash-linear-attention.git](https://github.com/fla-org/flash-linear-attention.git)
cd flash-linear-attention
pip install -e .
cd ..

# 2. Flame - The training engine
git clone [https://github.com/fla-org/flame.git](https://github.com/fla-org/flame.git)
cd flame
pip install -e .
cd ..

# 3. Palimpsa - This repository
git clone git@github.com:djo1996/Palimpsa.git
cd Palimpsa
pip install -e .
```

---

## 🚀 Usage: Training with Flame

Palimpsa is designed to be a "drop-in" extension for Flame. We provide a custom launcher (`train.py`) that registers the Palimpsa model architecture into the Flame registry automatically.

### 1. Configuration
Create a config file in `configs/`. You can inherit settings from standard Flame configs.

**Example:** `configs/palimpsa_340M.yaml`
```yaml
model:
  name: palimpsa        # Registered model name
  flavor: 340M          # Size flavor (matches paper experiments)
  expand_k: 0.5         # Palimpsa uses compressed state size for efficiency
  expand_v: 1.0
  
training:
  tensor_parallel_degree: 1       # TP=1
  data_parallel_shard_degree: 8   # FSDP across 8 GPUs
  # See flame/configs for scheduler/optimizer details
```

### 2. Launch Training
**Important:** Do not use the standard `flame` CLI. Use the `train.py` provided in this repo, which ensures the Palimpsa plugins are loaded before the engine starts.

```bash
# Ensure you are inside the Palimpsa/ directory
torchrun --nproc_per_node=8 train.py --config configs/palimpsa_340M.yaml
```

---

## 📊 Performance & Benchmarks

### Mechanistic Architecture Design (MAD)
Palimpsa achieves competitive scores on the MAD benchmark, excelling in state-tracking tasks.
- **Perfect Score (100%)** on *IC & Noisy Recall*.
- **Top-tier performance** on *Memorize* and *Selective Copy*.

### Language Modeling
Tested on **FineWeb-Edu** (15B/30B tokens):
- **170M / 340M parameters:** Palimpsa outperforms strong baselines like **Gated DeltaNet** and **Transformer++** on perplexity and zero-shot commonsense reasoning (HellSwag, PIQA, etc.).
- **Scalability:** Uses a fused chunk-wise parallel scan (Triton) to maintain high training throughput.

---

## 📂 Repository Structure

```text
Palimpsa_Lab/
├── flash-linear-attention/     # Upstream dependency (cloned)
├── flame/                      # Training engine (cloned)
└── Palimpsa/                   # This Repo
    ├── palimpsa/               
    │   ├── layers/             # PalimpsaLayer (Bayesian update rules)
    │   ├── models/             # HF-compatible Modeling code
    │   ├── ops/                # Fused Triton kernels for parallel scan
    │   └── integration.py      # Flame registry hooks
    ├── configs/                # Experiment configurations
    ├── train.py                # Custom launcher
    └── pyproject.toml
```

---

## 📜 Citation

If you use this codebase or the Palimpsa architecture in your research, please cite our paper:

```bibtex
@article{bonnet2025palimpsa,
  title={Learning to Remember, Learn, and Forget in Attention-Based Models},
  author={Bonnet, Djohan and et al.},
  journal={Under Review},
  year={2025},
  url={[https://github.com/djo1996/Palimpsa](https://github.com/djo1996/Palimpsa)}
}
```
