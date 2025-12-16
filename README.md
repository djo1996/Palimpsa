<div align="center">
<img width="600" alt="Palimpsa Logo" src="https://github.com/user-attachments/assets/7fa41f32-0976-42c9-8d32-2a602e56289f" />

# Palimpsa
### Learning to Remember, Learn, and Forget in Attention-Based Models

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-blue)](https://arxiv.org/abs/2504.13569)
[![Framework](https://img.shields.io/badge/Built%20On-Flame%20%26%20FLA-firebrick)](https://github.com/fla-org/flame)
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

**Palimpsa** is a novel attention mechanism that views In-Context Learning (ICL) as a continual learning problem. It introduces **Bayesian Metaplasticity** to transformer architectures—dynamically adjusting the plasticity of memory states based on their uncertainty.

---

## 📂 Repository Structure

The repository is organized to support both research benchmarking (Zoology style) and large-scale pretraining (Flame/Hugging Face style).
```text
Palimpsa/
├── benchmark_mqar.py       # Main entry point for MQAR benchmarks
├── config_mqar.py          # Configs for Palimpsa vs. Baselines (GLA, DeltaNet)
├── model_mqar.py           # Zoology-style backbone adapter for FLA layers
├── train_nano.py           # NanoGPT training script
├── palimpsa/               # Core package source code
│   ├── layers/             # PyTorch layers implementation
│   ├── models/             # Hugging Face compatible model definitions
│   ├── ops/                # Optimized CUDA/Triton kernels
│   ├── check_palimpsa.py   # Implementation sanity checks
│   └── integration.py      # Integration utilities
├── data/
│   └── data_mqar/          # Rigorous Zoology data generation pipeline
│       ├── associative_recall.py
│       └── config.py
└── ...
```
## 🛠️ Installation (Core)

This sets up the core environment required to run the model and the NanoGPT quick start.


### 1. Create Workspace
```bash
mkdir Palimpsa_Lab && cd Palimpsa_Lab

# Clone Palimpsa
git clone https://github.com/djo1996/Palimpsa.git

# Clone Dependencies
git clone https://github.com/fla-org/flash-linear-attention.git
```

### 2. Set Up Environment
We use `uv` for speed, but we bootstrap it inside a standard venv to avoid system conflicts.

```bash
# 1. Create and Activate a Standard Venv
python3 -m venv palimpsa_env
source palimpsa_env/bin/activate

# 2. Install uv inside the venv
pip install uv

# 4. Install Build Tools
uv pip install ninja packaging setuptools wheel

# 5. Install Kernels (From Source)
uv pip install causal-conv1d
uv pip install -e ./flash-linear-attention

# 6. Install Palimpsa
uv pip install -e ./Palimpsa
```

---

## 🚀 Quick Start: Shakespeare (NanoGPT)

Verify that the kernels are compiling and the model converges by training on the Shakespeare dataset.

```bash
cd Palimpsa

# 1. Prepare Data
python data/shakespeare_char/prepare.py

# 2. Train Palimpsa (Nano flavor)
python train_nano.py --model palimpsa --batch_size 16

# 3. Train Baselines (Optional)
# python train_nano.py --model gla --batch_size 16
# python train_nano.py --model gated_deltanet --batch_size 16
```
*You should see the loss dropping within the first few iterations.*

---

## 🔬 Advanced: Research Scale (Flame)

Follow these steps **only** if you want to train Large Language Models (LLMs) using the [Flame](https://github.com/fla-org/flame) engine.

### 1. Install Flame Engine
Return to the `Palimpsa_Lab` root directory.

```bash
cd .. 

# 1. Clone Flame
git clone https://github.com/fla-org/flame.git

# 2. Install TorchTitan (Specific commit required for FSDP)
uv pip install git + https://github.com/pytorch/torchtitan.git@0b44d4c

# 3. Install Flame
uv pip install -e ./flame
```

### 2. Download FineWeb-Edu
Flame requires the dataset to be cached locally.

```bash
# Run this from the Palimpsa directory
cd Palimpsa
python data/download_fineweb.py --cache_dir /Local/your_name/.cache
```

### 3. Launch Training (Slurm)
Use `torchrun` via Slurm. Ensure your script exports the same CUDA variables as the installation.

**Example `train.slurm`:**
```bash
#!/bin/bash
#SBATCH --job-name=Palimpsa
#SBATCH --partition=pgi15-h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00

source palimpsa_env/bin/activate

# CRITICAL: Match Install Environment
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    Palimpsa/train.py \
    --job.config_file flame/flame/models/fla.toml \
    --model.name palimpsa \
    --model.config Palimpsa/configs/palimpsa_170M.json \
    --training.dataset_name sample-100BT
```

---

## 📊 Benchmarks

### Language Modeling (FineWeb-Edu)
- **170M / 340M parameters:** Palimpsa outperforms strong baselines like **Gated DeltaNet** and **Transformer++** on perplexity and zero-shot commonsense reasoning (HellSwag, PIQA).
- **Scalability:** Uses a fused chunk-wise parallel scan (Triton) to maintain high training throughput.

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
