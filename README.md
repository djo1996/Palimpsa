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

## 1. Installation

For a quick start, you can install the standalone package.

```bash
# 1. Create env (Python 3.10+ recommended)
conda create -n palimpsa_env python=3.10
conda activate palimpsa_env

# 2. Install dependencies (We need Triton for the kernels)
pip install torch packaging ninja
pip install -U flash-attn --no-build-isolation

# 3. Install Flash Linear Attention (FLA) & Palimpsa
pip install git+[https://github.com/fla-org/flash-linear-attention.git](https://github.com/fla-org/flash-linear-attention.git)
git clone [https://github.com/djo1996/Palimpsa.git](https://github.com/djo1996/Palimpsa.git)
cd Palimpsa
pip install -e .
```

---

## 2. Test Run: Shakespeare (NanoGPT Style)

Before scaling up, you can verify that the kernels are compiling and the model converges by training on the Shakespeare dataset (character-level). This codebase is adapted to support a lightweight "NanoGPT" style loop for debugging.

**1. Prepare the data:**
```bash
cd data/shakespeare
python prepare.py
cd ../..
```

**2. Train Palimpsa:**
```bash
# Trains a small Palimpsa model on a single GPU
python train_nano.py --model palimpsa --batch_size 64 --compile
```
*You should see the loss dropping within the first few iterations.*

---

## 3. Research Scale: Training with Flame

To reproduce the paper results (170M/340M models on FineWeb-Edu) or to train at scale, we use the **[🔥 Flame](https://github.com/fla-org/flame)** engine. We recommend setting up a dedicated workspace named `Palimpsa_Lab` to handle the specific versioning required for the bleeding-edge stack.

### Setup Workspace
```bash
# 1. Create the lab
mkdir Palimpsa_Lab && cd Palimpsa_Lab

# 2. Install specific TorchTitan commit (Required by Flame)
pip install git+[https://github.com/pytorch/torchtitan.git@0b44d4c](https://github.com/pytorch/torchtitan.git@0b44d4c)

# 3. Install Flame (Training Engine)
git clone [https://github.com/fla-org/flame.git](https://github.com/fla-org/flame.git)
cd flame && pip install -e . && cd ..

# 4. Link Palimpsa
# (Assuming you cloned Palimpsa inside Palimpsa_Lab, or symlink it here)
```

### Launch Training
Palimpsa acts as a plugin for Flame. Use the provided launcher to register the architecture:

```bash
# Example: 340M model on 8 GPUs
torchrun --nproc_per_node=8 train.py --config configs/palimpsa_340M.yaml
```

---

## 📊 Performance

### Mechanistic Architecture Design (MAD)
Palimpsa achieves competitive scores on the MAD benchmark, excelling in state-tracking tasks.
- **Perfect Score (100%)** on *IC & Noisy Recall*.
- **Top-tier performance** on *Memorize* and *Selective Copy*.

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
