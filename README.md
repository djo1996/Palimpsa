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

## 🛠️ Installation: Core Components

We recommend setting up a unified workspace. This installation covers the core model and kernels required to run small-scale experiments (like Shakespeare).

### 1. Create Workspace & Environment
```bash
# 1. Create the working directory
mkdir Palimpsa_Lab
cd Palimpsa_Lab

# 2. Create and activate a virtual environment
# (Using Python 3.10+ is recommended)
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install Build Dependencies
To compile the custom CUDA kernels, we need PyTorch and build tools installed first.

```bash
pip install torch packaging ninja
```

### 3. Install Core Kernels & Palimpsa
We install from source to ensure CUDA kernels match your local setup.

```bash
# 1. Causal Conv1d
git clone git@github.com:Dao-AILab/causal-conv1d.git
pip install ./causal-conv1d

# 2. Flash Linear Attention (FLA)
git clone [https://github.com/fla-org/flash-linear-attention.git](https://github.com/fla-org/flash-linear-attention.git)
pip install -e ./flash-linear-attention

# 3. Palimpsa (This Repo)
git clone [https://github.com/djo1996/Palimpsa.git](https://github.com/djo1996/Palimpsa.git)
pip install -e ./Palimpsa
```

## 🚀 Quick Start: Shakespeare (NanoGPT)

Before launching large-scale runs, verify that the kernels are compiling and the model converges by training on the Shakespeare dataset.

```bash
# 1. Prepare data
cd Palimpsa/data/shakespeare_char
python prepare.py
cd ../../..  # Return to Palimpsa_Lab root

# 2. Train Palimpsa (Nano flavor)
# Note: Execute from Palimpsa root
cd Palimpsa
python train_nano.py --model palimpsa --batch_size 16
```
*You should see the loss dropping within the first few iterations.*

---

## 🔬 Advanced: Research Scale (Flame)

**Note:** Only follow this step if you intend to train Large Language Models (LLMs) using the [🔥 Flame](https://github.com/fla-org/flame) engine. This requires a specific FSDP stack.

### Install Flame Engine
```bash
# 1. Install specific TorchTitan commit (Required for FSDP support in Flame)
pip install git+[https://github.com/pytorch/torchtitan.git@0b44d4c](https://github.com/pytorch/torchtitan.git@0b44d4c)

# 2. Install Flame
# Assuming you are in Palimpsa_Lab root
git clone [https://github.com/fla-org/flame.git](https://github.com/fla-org/flame.git)
pip install -e ./flame
```

### Prepare FineWeb-Edu Data
Training with Flame requires the dataset to be cached locally on the machine.

**1. Create the download script:**
(This is already provided in `data/download_fineweb.py`)

**2. Run the download:**
Replace `/Local/your_name/.cache` with a path where you have significant storage space (approx. 500GB for 100BT tokens).

```bash
python data/download_fineweb.py --cache_dir /Local/your_name/.cache
```

### Launch Training (FineWeb-Edu)
To reproduce the paper results (170M/340M models), use the `train.py` launcher inside the `Palimpsa/` directory.

```bash
cd Palimpsa
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

## 🙏 Acknowledgements

This project stands on the shoulders of giants. We adapted code and drew inspiration from these excellent repositories:

* **[Longhorn](https://github.com/Cranial-XIX/longhorn):** For the sleek NanoGPT-style training loop and inspiration on online learning in SSMs.
* **[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention):** For the foundational linear attention implementations and chunk-wise parallel forms.
* **[Flame](https://github.com/fla-org/flame):** For the robust training engine and FSDP integration.
* **[Mamba](https://github.com/state-spaces/mamba/tree/main):** For the foundational SSM modeling components and high-performance selective scan kernels.
* **[Zoology](https://github.com/HazyResearch/zoology):** For synthetic task design and evaluation protocols.
* **[MAD Lab](https://github.com/athms/mad-lab):** For the mechanistic interpretability and synthetic recall benchmarks.

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
