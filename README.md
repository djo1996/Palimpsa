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

---

## 🛠️ Installation (The "Zero to Hero" Setup)

We recommend setting up a unified workspace. This guide uses **`uv`** for fast dependency resolution and ensures compatibility with modern hardware (H100/H200).

> [!IMPORTANT]
> **Compile where you run!**
> This installation involves compiling custom CUDA kernels (`flash-linear-attention`, `causal-conv1d`).
> You **MUST** run these steps on a **GPU Compute Node** (e.g., H100), not a login node.
>
> **Interactive Session:** `srun --partition=your-h100-partition --gres=gpu:1 --pty bash`

### 1. Create Workspace & Clone Repos
We organize everything into a `Palimpsa_Lab` directory to keep dependencies (Flame, FLA) side-by-side.

```bash
mkdir Palimpsa_Lab && cd Palimpsa_Lab

# 1. Clone Palimpsa
git clone [https://github.com/djo1996/Palimpsa.git](https://github.com/djo1996/Palimpsa.git)

# 2. Clone Dependencies
git clone [https://github.com/fla-org/flame.git](https://github.com/fla-org/flame.git)
git clone [https://github.com/fla-org/flash-linear-attention.git](https://github.com/fla-org/flash-linear-attention.git)
```

### 2. Install Environment
We use `uv` to speed up the process and **PyTorch Nightly** to ensure compatibility with Flame and H100 drivers.

```bash
# 1. Install uv (Lightning fast pip replacement)
pip install uv

# 2. Create Virtual Env
uv venv palimpsa_env
source palimpsa_env/bin/activate

# 3. Load System CUDA (Crucial for H100s)
# Adjust 'module load' for your specific cluster
module load CUDA 
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 4. Install PyTorch Nightly (Required for Flame)
# We target CUDA 12.6 to match modern drivers
uv pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu126](https://download.pytorch.org/whl/nightly/cu126)

# 5. Install Build Tools
uv pip install ninja packaging setuptools wheel

# 6. Install Kernels (From Source)
# --no-cache-dir ensures binaries are built for YOUR specific GPU
uv pip install --no-cache-dir causal-conv1d
uv pip install --no-cache-dir -e ./flash-linear-attention

# 7. Install Flame Engine
# Requires specific torchtitan commit for FSDP stability
uv pip install git+[https://github.com/pytorch/torchtitan.git@0b44d4c](https://github.com/pytorch/torchtitan.git@0b44d4c)
uv pip install -e ./flame

# 8. Install Palimpsa
uv pip install --no-cache-dir -e ./Palimpsa
```

### 3. Verification
Run this one-liner to verify that PyTorch is talking to your GPU correctly:
```bash
python -c "import torch; print(f'⚡ Torch: {torch.__version__}'); print(f'🚀 CUDA Available: {torch.cuda.is_available()}'); import fla; print('✅ FLA Loaded Successfully')"
```

---

## 🚀 Quick Start: Shakespeare (NanoGPT)

For quick debugging on a single GPU without the overhead of the Flame engine.

```bash
cd Palimpsa

# 1. Prepare Data
python data/shakespeare_char/prepare.py

# 2. Train Palimpsa
python train_nano.py --model palimpsa --batch_size 16

# 3. Train Baselines
# python train_nano.py --model gla --batch_size 16
# python train_nano.py --model gated_deltanet --batch_size 16
```

---

## 🔬 Research Scale Training (Flame)

To train large models (170M+) using FSDP and the Flame engine.

### 1. Download FineWeb-Edu
Flame requires the dataset to be cached locally.

```bash
# Run from Palimpsa_Lab/Palimpsa/
python data/download_fineweb.py --cache_dir /Local/your_name/.cache
```

### 2. Launch Training (Slurm)
We use `torchrun` via Slurm. Ensure your script exports the same CUDA variables as the installation.

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

### Synthetic Tasks (MAD / MQAR)
Benchmarks for Mechanistic Architecture Design (MAD) and Multi-Query Associative Recall (MQAR) are **coming soon**.
*Preliminary results show perfect scores on Noisy Recall and competitive performance on State Tracking.*

---

## 🙏 Acknowledgements

This project stands on the shoulders of giants:

* **[Longhorn](https://github.com/Cranial-XIX/longhorn):** NanoGPT-style training loop.
* **[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention):** Foundational linear attention kernels.
* **[Flame](https://github.com/fla-org/flame):** Training engine and FSDP integration.
* **[Zoology](https://github.com/HazyResearch/zoology):** Synthetic task evaluation.

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
