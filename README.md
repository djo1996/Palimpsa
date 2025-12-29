<div align="center">
<img width="600" alt="Palimpsa Logo" src="https://github.com/user-attachments/assets/7fa41f32-0976-42c9-8d32-2a602e56289f" />

# Palimpsa
### Learning to Remember, Learn, and Forget in Attention-Based Models

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-blue)](https://arxiv.org/abs/2504.13569)
[![Framework](https://img.shields.io/badge/Built%20On-Zoology%20%26%20FLA-firebrick)](https://github.com/hazy-research/zoology)
[![License](https://img.shields.io/badge/License-MIT-green)]()

</div>

**Palimpsa** is a novel attention mechanism that views In-Context Learning (ICL) as a continual learning problem. It introduces **Bayesian Metaplasticity** to transformer architectures—dynamically adjusting the plasticity of memory states based on their uncertainty.

---

## 📂 Repository Structure

The repository is structured to handle both synthetic associative recall benchmarks (Zoology) and optimized linear attention kernels (FLA).

```text
Palimpsa/
├── fla/                    # Optimized CUDA/Triton kernels for Palimpsa
├── zoology/                # Research framework for associative recall
│   ├── mqar_figure/        # Sweep configurations for MQAR benchmarks
│   ├── mixers/             # Palimpsa & GatedDeltaNet layer wrappers
│   ├── launch.py           # Main entry point for running sweeps
│   └── train.py            # Core training loop logic
├── cache/                  # Local directory for generated MQAR datasets
└── ...
```

## 🛠️ Installation & Environment

### 1. Shell Configuration (W&B)
To ensure the logger works across both private clusters (like FZ-Juelich) and public clouds, add these to your `~/.bashrc`. This keeps your API keys and private URLs out of the codebase.

```bash
# Add to the end of your ~/.bashrc
export WANDB_API_KEY="your_key_here"
export WANDB_BASE_URL="[https://wandb.fz-juelich.de](https://wandb.fz-juelich.de)"  # Or [https://api.wandb.ai](https://api.wandb.ai)
export WANDB_ENTITY="your_username"

# Then reload your shell
source ~/.bashrc
```

### 2. Set Up Environment
```bash
# 1. Create and Activate a Standard Venv
python3 -m venv palimpsa_env
source palimpsa_env/bin/activate

# 2. Install core dependencies
pip install -e .
```

---

## 🚀 Running MQAR Benchmarks

Palimpsa is integrated with the Zoology launch system. You can run individual sweeps or full benchmarks using the following command structure:

```bash
# From the project root
python3 -m zoology.launch zoology/mqar_figure/configs.py
```

### Reproducing Figure 2 (MQAR)
The `configs.py` file contains the hyperparameters for benchmarking Palimpsa against Gated Delta Networks (GDN) and other baselines. 
- **Sequence Lengths:** 512, 1024
- **Model Dimension:** 128
- **Learning Rate:** 0.01

---

## 🔬 Advanced: Research Scale (Flame)

Palimpsa is also compatible with the [Flame](https://github.com/fla-org/flame) engine for large-scale pretraining.

### Launch Training (Slurm)
For H100/A100 clusters using Slurm, ensure your environment variables are correctly exported in your batch script before running `torchrun`.

```bash
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    Palimpsa/train.py \
    --model.name palimpsa \
    --training.seq_len 32768 \
    --training.dataset HuggingFaceFW/fineweb-edu
```

---

## 📜 Citation

If you use this codebase or the Palimpsa architecture in your research, please cite:

```bibtex
@article{bonnet2025palimpsa,
  title={Learning to Remember, Learn, and Forget in Attention-Based Models},
  author={Bonnet, Djohan and et al.},
  journal={Under Review},
  year={2025},
  url={[https://github.com/djo1996/Palimpsa](https://github.com/djo1996/Palimpsa)}
}
```
