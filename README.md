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

## 🛠️ Installation: The "Palimpsa_Lab"

To ensure compatibility between the kernels, the training engine, and the model, we recommend setting up a unified workspace. This allows you to easily modify the upstream `FLA` kernels or `Flame` engine if needed.

### 1. Create Workspace & Environment
Start by creating the directory and the virtual environment.

```bash
# 1. Create the working directory
mkdir Palimpsa_Lab
cd Palimpsa_Lab

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install Core Kernels
We need to compile `causal-conv1d` from source to ensure the CUDA kernels match your local setup.

```bash
git clone git@github.com:Dao-AILab/causal-conv1d.git
cd causal_conv1d
python setup.py install
cd ..
```

### 3. Install Research Stack
Install the libraries in this order. We use **editable mode** (`-e .`) so changes you make to the code are immediately reflected.

```bash
# 1. Flash Linear Attention (FLA)
git clone [https://github.com/fla-org/flash-linear-attention.git](https://github.com/fla-org/flash-linear-attention.git)
cd flash-linear-attention
pip install -e .
cd ..

# 2. Flame (Training Engine)
# Note: Flame requires specific torch-titan commits for FSDP support
pip install git+[https://github.com/pytorch/torchtitan.git@0b44d4c](https://github.com/pytorch/torchtitan.git@0b44d4c)
git clone [https://github.com/fla-org/flame.git](https://github.com/fla-org/flame.git)
cd flame
pip install -e .
cd ..

# 3. Palimpsa (This Repo)
git clone git@github.com:djo1996/Palimpsa.git
cd Palimpsa
pip install -e .
```

---

## 🚀 Quick Start: Shakespeare (NanoGPT)

Before launching large-scale runs, verify that the kernels are compiling and the model converges by training on the Shakespeare dataset.

```bash
# 1. Prepare data
cd data/shakespeare
python prepare.py
cd ../..

# 2. Train Palimpsa (Nano flavor)
python train_nano.py --model palimpsa --batch_size 64 --compile
```
*You should see the loss dropping within the first few iterations.*

---

## 🔬 Research Scale: Training with Flame

To reproduce the paper results (170M/340M models on FineWeb-Edu), use the `train.py` launcher which registers Palimpsa into the Flame registry.

```bash
# Run from inside the Palimpsa/ directory
torchrun --nproc_per_node=8 train.py --config configs/palimpsa_340M.yaml
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
