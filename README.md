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

The repository is organized to support both MQAR research benchmarks and large-scale pretraining using the Flame engine.

```text
Palimpsa_Lab/
├── Palimpsa/               # Main Research Repo
│   ├── palimpsa/           # Core library (layers, models)
│   │   └── integration.py  # Flame/Torch-Titan plugin registry
│   ├── config/             # Model architecture JSONs
│   ├── evaluation/         # Evaluation Harness (NEW)
│   │   ├── launcher.py     # Python entry point for lm-eval
│   │   └── run_eval.sh     # Bash dispatcher for experiments
│   ├── tools/              # Utilities (NEW)
│   │   └── convert_dcp_to_hf.py # Checkpoint converter
│   ├── zoology/            # MQAR/Associative recall benchmarks
│   └── train.py            # Unified training entry point
├── flame/                  # Training engine (submodule/clone)
└── flash-linear-attention/ # Fused kernels (submodule/clone)
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

Verify model convergence on the Shakespeare dataset. Logic adapted from [Longhorn](https://github.com/Cranial-XIX/longhorn).
```bash
cd Palimpsa
python data/shakespeare_char/prepare.py
python train_nano.py --model palimpsa --batch_size 16
```

---

## 📊 MQAR Benchmarking (Zoology)

Reproduce Multi-Query Associative Recall (MQAR) results using the [Zoology](https://github.com/HazyResearch/zoology) repository.
```bash
# Run the MQAR figure sweep
python3 -m zoology.launch zoology/mqar_figure/configs.py --name palimpsa_sweep
# If you have severals GPUS your nodes
python3 -m zoology.launch zoology/mqar_figure/configs.py --gpus 0,1,2,3 --name palimpsa_sweep
```
*Datasets are automatically generated and stored in the local `cache/` directory.*

---

## Meta HMM
#In the Palimpsa directory, run
pip install palimpsa[metahmm]
python train_nano_hmm.py --model palimpsa --batch_size 128

---

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
#### 1. Configure Environment
Set your cluster-specific paths and W&B credentials:
```bash
export HF_DATASETS_CACHE="/path/to/your/cache"
export WANDB_PROJECT="Palimpsa"
```

#### 2. Launching via Torchrun
You can run the training directly or via an interactive session.

```bash
torchrun --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    Palimpsa/train.py \
    --job.config_file flame/flame/models/fla.toml \
    --job.dump_folder exp/palimpsa-170M \
    --model.name palimpsa \
    --model.config Palimpsa/config/palimpsa_170M.json \
    --model.tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --optimizer.lr 3e-3 \
    --lr_scheduler.warmup_steps 2000 \
    --training.batch_size 1 \
    --training.gradient_accumulation_steps 4 \
    --training.seq_len 32768 \
    --training.context_len 4096 \
    --training.varlen \
    --training.steps 3000 \
    --training.dataset HuggingFaceFW/fineweb-edu \
    --training.dataset_name sample-100BT \
    --training.dataset_split train \
    --training.num_workers 8 \
    --checkpoint.interval 200 \
    --metrics.log_freq 10
```
## ⚖️ Model Evaluation

We provide a robust pipeline to convert distributed checkpoints and run standard benchmarks using `lm-evaluation-harness`.

### 1. Convert Checkpoint
Training produces Distributed Checkpoints (DCP). Before evaluating, convert them to Hugging Face format. This command automatically finds the config and tokenizer snapped during training.

```bash
# Example: Convert step 3000 of the 'palimpsa-170M' experiment
python tools/convert_dcp_to_hf.py --exp ../exp/palimpsa-170M --step 3000
```
*Outputs to: `../exp/palimpsa-170M/hf_model_step_3000/`*

### 2. Run Benchmarks
Use the dispatcher script to launch evaluation on a specific GPU. This handles path setup and logging automatically.

**Usage:**
```bash
bash evaluation/run_eval.sh [GPU_ID] [MODEL_NAME] [STEP] [TASKS] [EXTRA_ARGS]
```

**Examples:**

Run standard benchmarks on GPU 0:
```bash
bash evaluation/run_eval.sh 0 palimpsa-170M 3000 "wikitext,hellaswag,piqa"
```

Run advanced configuration (few-shot, limit samples) on GPU 3:
```bash
bash evaluation/run_eval.sh 3 palimpsa-170M 3000 "lambada_openai" --num_fewshot 5 --batch_size 8 --limit 100
```

**Parallel Evaluation on Multiple GPUs:**
You can launch multiple evaluations simultaneously by specifying different GPU IDs:
```bash
# Run wikitext on GPU 0
bash evaluation/run_eval.sh 0 palimpsa-170M 3000 "wikitext" &

# Run hellaswag on GPU 1
bash evaluation/run_eval.sh 1 palimpsa-170M 3000 "hellaswag" &

wait
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
