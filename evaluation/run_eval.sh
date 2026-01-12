#!/bin/bash
# Usage: ./run_eval.sh [GPU_ID] [MODEL_NAME]
# Example: ./run_eval.sh 0 palimpsa-170M

GPU_ID=${1:-0}
MODEL_NAME=${2:-"palimpsa-170M"}

# Paths
ROOT_DIR="$(pwd)" # Assume running from Palimpsa/ root
EXP_DIR="${ROOT_DIR}/../exp/${MODEL_NAME}" # Adjust based on your folder structure
DCP_PATH="${EXP_DIR}/checkpoint/step-3000" # You might want to paramterize the step
HF_OUT_PATH="${EXP_DIR}/hf_model"
CONFIG_PATH="${ROOT_DIR}/config/palimpsa_170M.json"
TOKENIZER="meta-llama/Llama-2-7b-chat-hf"

# 1. Check if conversion is needed
if [ ! -d "$HF_OUT_PATH" ]; then
    echo "⚙️  HF Model not found. Converting DCP to HF..."
    python tools/convert_dcp.py \
        --dcp "$DCP_PATH" \
        --out "$HF_OUT_PATH" \
        --config "$CONFIG_PATH" \
        --tokenizer "$TOKENIZER"
else
    echo "✅ HF Model found at $HF_OUT_PATH. Skipping conversion."
fi

# 2. Run Evaluation
echo "🧪 Starting Evaluation on GPU ${GPU_ID}..."
export CUDA_VISIBLE_DEVICES=$GPU_ID

python evaluation/launcher.py \
    --model_path "$HF_OUT_PATH" \
    --tasks "wikitext,hellaswag" \
    --batch_size 16 \
    --output_path "${EXP_DIR}/eval_results.json"