#!/bin/bash
# Usage: bash examples/scripts/run_eval.sh <path_to_hf_checkpoint> <tasks>

# 1. Arguments & Defaults
MODEL_PATH=${1:-"exp/palimpsa-170M-100BT/hf_model"} # Default path
TASKS=${2:-"hellaswag,piqa,arc_easy"}               # Default tasks



# WandB Setup (Optional)
export WANDB_PROJECT="Palimpsa-Eval"
export WANDB_JOB_TYPE="eval"
# export WANDB_API_KEY="..." # Better to have this in ~/.bashrc

echo "------------------------------------------------"
echo "🔍 Evaluating Model: $MODEL_PATH"
echo "📋 Tasks: $TASKS"
echo "------------------------------------------------"

# 3. Run Evaluation
# We use 'hf' model type, but pass 'trust_remote_code' just in case.
# Because we import palimpsa/integration.py inside evaluate.py, 
# 'pretrained=$MODEL_PATH' will automatically resolve the 'palimpsa' architecture.

python Palimpsa/evaluate.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True \
    --tasks $TASKS \
    --batch_size auto \
    --device cuda:0 \
    --output_path results/$(basename $MODEL_PATH)