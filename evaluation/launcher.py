import argparse
import sys
from lm_eval import simple_evaluate
from lm_eval.utils import make_table
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- 1. IMPORT YOUR MODELS ---
from palimpsa.models.palimpsa import PalimpsaConfig, PalimpsaForCausalLM
from palimpsa.models.meta_mamba2 import MetaMamba2Config, MetaMamba2ForCausalLM

# --- 2. REGISTER THEM ---
# This tells HF: "When you see 'palimpsa' or 'meta_mamba2', use THESE classes."
AutoConfig.register("palimpsa", PalimpsaConfig)
AutoModelForCausalLM.register(PalimpsaConfig, PalimpsaForCausalLM)

AutoConfig.register("meta_mamba2", MetaMamba2Config)
AutoModelForCausalLM.register(MetaMamba2Config, MetaMamba2ForCausalLM)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HF converted model")
    parser.add_argument("--tasks", type=str, default="wikitext", help="Comma separated tasks")
    parser.add_argument("--batch_size", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    print(f"🚀 Launching Eval for: {args.model_path}")
    print(f"📋 Tasks: {args.tasks}")

    # --- 3. CRITICAL FIX: trust_remote_code=False ---
    # We must set this to False so it uses the classes we registered above,
    # instead of trying to find the code specified in config.json's 'auto_map'.
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model_path},trust_remote_code=False,dtype=bfloat16",
        tasks=args.tasks.split(","),
        batch_size=args.batch_size,
        device=args.device
    )

    if results:
        print(make_table(results))
        if args.output_path:
            import json
            # Helper to convert non-serializable types if needed
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()