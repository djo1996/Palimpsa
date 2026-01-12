#!/bin/bash
# Usage: ./evaluation/run_eval.sh [GPU_ID] [MODEL_NAME] [STEP]
# Example: ./evaluation/run_eval.sh 0 meta_mamba2-170M 3000

GPU_ID=${1:-0}
MODEL_NAME=${2:-"palimpsa-170M"}
STEP=${3:-3000}

# Paths
ROOT_DIR="$(pwd)" 
EXP_DIR="${ROOT_DIR}/../exp/${MODEL_NAME}"
HF_OUT_PATH="${EXP_DIR}/hf_model_step_${STEP}"

# 1. Safety Check: Ensure Model is Converted
if [ ! -d "$HF_OUT_PATH" ]; then
    echo "❌ Error: HF Model not found at ${HF_OUT_PATH}"
    echo "---------------------------------------------------"
    echo "Please convert the checkpoint first by running:"
    echo ""
    echo "   python tools/convert_dcp_to_hf.py --exp ${EXP_DIR} --step ${STEP}"
    echo ""
    echo "---------------------------------------------------"
    exit 1
fi

# 2. Run Evaluation
echo "🧪 Starting Evaluation on GPU ${GPU_ID}..."
echo "   Model: ${MODEL_NAME} (Step ${STEP})"
echo "   Path:  ${HF_OUT_PATH}"

export CUDA_VISIBLE_DEVICES=$GPU_ID

python evaluation/launcher.py \
    --model_path "$HF_OUT_PATH" \
    --tasks "wikitext,hellaswag" \
    --batch_size 16 \
    --output_path "${EXP_DIR}/eval_results_step_${STEP}.json"

echo "✅ Done. Results saved to ${EXP_DIR}/eval_results_step_${STEP}.json"