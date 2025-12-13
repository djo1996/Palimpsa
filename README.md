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

## 🛠️ Installation (Core)

This sets up the core environment required to run the model and the NanoGPT quick start.

> [!IMPORTANT]
> **Compile where you run!**
> You **MUST** run these steps on a **GPU Compute Node** (e.g., H100), not a login node.
>
> **Interactive Session:** `srun --partition=pgi15-h100 --gres=gpu:1 --pty bash`

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

# 3. Load System CUDA (Crucial for H100s)
# Note: If 'module' command is not found, ensure you are on a compute node or skip this line.
# module load CUDA
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4. Install PyTorch Nightly (Required for Flame/H100)
# We target CUDA 12.6 to match modern drivers.
uv pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu126](https://download.pytorch.org/whl/nightly/cu126)

# 5. Install Build Tools
uv pip install ninja packaging setuptools wheel

# 6. Install Kernels (From Source)
# --no-cache-dir ensures binaries are built for YOUR specific GPU
uv pip install --no-cache-dir causal-conv1d
uv pip install --no-cache-dir -e ./flash-linear-attention

# 7. Install Palimpsa
uv pip install --no-cache-dir -e ./Palimpsa
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
git clone [https://github.com/fla-org/flame.git](https://github.com/fla-org/flame.git)

# 2. Install TorchTitan (Specific commit required for FSDP)
uv pip install git+[https://github.com/pytorch/torchtitan.git@0b44d4c](https://github.com/pytorch/torchtitan.git@0b44d4c)

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
