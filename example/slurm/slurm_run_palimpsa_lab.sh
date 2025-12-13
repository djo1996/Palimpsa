#!/bin/bash
#SBATCH --job-name=Palimpsa-760M
#SBATCH --error=runs/%x_%j.err
#SBATCH --output=runs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # Torchrun is 1 task per node
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=100:00:00
#SBATCH --partition=pgi15

# =========================================================
# 1. Environment & Secrets
# =========================================================
# Assumes the script is submitted from the project root (Palimpsa_Lab)
source .venv/bin/activate

# CACHE CONFIG
# Ensure this points to your fast local storage
export HF_DATASETS_CACHE="/Local/$USER/.cache"

# WANDB CONFIG
# Best practice: Don't hardcode keys. Export this in your shell or .bashrc
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY is not set. Logging might fail."
fi
export WANDB_PROJECT="Palimpsa"
export WANDB_WATCH="false"

# =========================================================
# 2. Distributed Config
# =========================================================
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export RDZV_ID=$SLURM_JOB_ID

echo "🚀 Launching on Node: $MASTER_ADDR"

# =========================================================
# 3. Training Launch
# =========================================================
# Note: We override --model.name to "palimpsa" to ensure it uses
# the custom TrainSpec registered in palimpsa/integration.py

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_id=$RDZV_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    Palimpsa/train.py \
    --job.config_file flame/models/fla.toml \
    --job.dump_folder exp/palimpsa-170M-100BT \
    --model.name palimpsa \
    --model.config Palimpsa/configs/palimpsa_170M.json \
    --model.tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --optimizer.lr 1.25e-3 \
    --lr_scheduler.warmup_steps 2000 \
    --training.batch_size 1 \
    --training.gradient_accumulation_steps 2 \
    --training.seq_len 32768 \
    --training.context_len 4096 \
    --training.varlen \
    --training.steps 60100 \
    --training.dataset HuggingFaceFW/fineweb-edu \
    --training.dataset_name sample-100BT \
    --training.dataset_split train \
    --training.num_workers 8 \
    --checkpoint.interval 2000 \
    --metrics.log_freq 10