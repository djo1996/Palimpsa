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

The repository supports research benchmarking (Zoology style) and large-scale pretraining (Flame/Hugging Face style).

```text
Palimpsa/
├── zoology/                # Research framework for associative recall
│   ├── mqar_figure/        # Sweep configs for MQAR benchmarks
│   ├── mixers/             # Palimpsa & GatedDeltaNet adapters
│   ├── launch.py           # Sweep entry point
│   └── train.py            # Training loop logic
├── palimpsa/               # Core package source code
│   ├── layers/             # PyTorch layers implementation
│   ├── models/             # Hugging Face compatible models
│   └── integration.py      # Flame engine integration
├── data/                   # Data preparation scripts
└── cache/                  # Generated datasets (local)
```

## 🛠️ Installation

### 1. Environment & W&B Setup
To keep the code clean and cluster-agnostic, configure your environment variables in your `~/.bashrc`. This ensures the logger works on private clusters without code changes.

```bash
# Add to ~/.bashrc
export WANDB_API_KEY=your_key_here
export WANDB_BASE_URL=your_cluser_url # Or https://api.wandb.ai
export WANDB_ENTITY=your_username

# Reload shell
source ~/.bashrc
```

### 2. Workspace & Dependencies
We use `uv` for high-speed dependency management inside a standard virtual environment.

```bash
mkdir Palimpsa_Lab && cd Palimpsa_Lab

# Clone Projects
git clone https://github.com/djo1996/Palimpsa.git
git clone https://github.com/fla-org/flash-linear-attention.git

# Set Up Venv
python3 -m venv palimpsa_env
source palimpsa_env/bin/activate
pip install uv

# Install Build Tools & Kernels
uv pip install ninja packaging setuptools wheel
uv pip install causal-conv1d
uv pip install -e ./flash-linear-attention
uv pip install -e ./Palimpsa
```

---

## 🚀 Quick Start: Shakespeare (NanoGPT)

Verify kernel compilation and model convergence on the Shakespeare character-level dataset.

```bash
cd Palimpsa
python data/shakespeare_char/prepare.py
python train_nano.py --model palimpsa --batch_size 16
```

---

## 📊 MQAR Benchmarking (Zoology)

Reproduce the Multi-Query Associative Recall (MQAR) results using the Zoology-integrated sweep system.

```bash
# Run the MQAR figure sweep
python3 -m zoology.launch zoology/mqar_figure/configs.py
```
*Datasets are automatically generated and stored in the local `cache/` directory.*

---

## 🔬 Advanced: Research Scale (Flame)

To train Large Language Models (LLMs) using the [Flame](https://github.com/fla-org/flame) engine:

### 1. Install Flame Engine
```bash
# Ensure you are in Palimpsa_Lab, NOT the Palimpsa repo
cd ..
```
```bash
git clone https://github.com/fla-org/flame.git
uv pip install git+https://github.com/pytorch/torchtitan.git@0b44d4c
uv pip install -e ./flame
```
### 2. Download FineWeb-Edu
Flame requires the dataset to be cached locally. Do it only once, preferably in sinteractive (faster)

```bash
# Run this from the Palimpsa directory
cd Palimpsa
python data/download_fineweb.py --cache_dir /Local/your_name/.cache
```

### 3. Launch Training (Slurm)
Ensure your Slurm script exports the necessary environment variables. The logger will automatically pick up your `WANDB_ENTITY` and `WANDB_BASE_URL`.

```bash
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    Palimpsa/train.py \
    --job.config_file flame/flame/models/fla.toml \
    --model.name palimpsa \
    --model.config Palimpsa/configs/palimpsa_170M.json \
    --training.dataset HuggingFaceFW/fineweb-edu \
    --training.seq_len 32768
```

---

## 📜 Citation

```bibtex
@article{bonnet2025palimpsa,
  title={Learning to Remember, Learn, and Forget in Attention-Based Models},
  author={Bonnet, Djohan and et al.},
  journal={Under Review},
  year={2025},
  url={[https://github.com/djo1996/Palimpsa](https://github.com/djo1996/Palimpsa)}
}
```
