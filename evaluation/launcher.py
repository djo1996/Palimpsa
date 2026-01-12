import argparse
import sys
import json
from lm_eval import simple_evaluate
from lm_eval.utils import make_table
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- 1. IMPORT YOUR MODELS ---
from palimpsa.models.palimpsa import PalimpsaConfig, PalimpsaForCausalLM
from palimpsa.models.meta_mamba2 import MetaMamba2Config, MetaMamba2ForCausalLM

# --- 2. REGISTER THEM ---
AutoConfig.register("palimpsa", PalimpsaConfig)
AutoModelForCausalLM.register(PalimpsaConfig, PalimpsaForCausalLM)

AutoConfig.register("meta_mamba2", MetaMamba2Config)
AutoModelForCausalLM.register(MetaMamba2Config, MetaMamba2ForCausalLM)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the HF converted model")
    parser.add_argument("--tasks", type=str, default="wikitext", help="Comma separated tasks")
    parser.add_argument("--batch_size", type=str, default="16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default=None)
    
    # --- NEW ARGUMENTS ---
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=float, default=None, help="Limit number of samples per task (for debugging)")
    parser.add_argument("--metadata", type=str, default=None, help="JSON string for extra metadata/config")

    args = parser.parse_args()

    print(f"🚀 Launching Eval for: {args.model_path}")
    print(f"📋 Tasks: {args.tasks}")
    print(f"🎯 Fewshot: {args.num_fewshot}, Batch: {args.batch_size}")

    # Parse metadata from JSON string if provided
    metadata_dict = None
    if args.metadata:
        try:
            metadata_dict = json.loads(args.metadata)
            print(f"ℹ️  Loaded Metadata: {metadata_dict}")
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parsing metadata JSON: {e}")

    # Run Evaluation
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model_path},trust_remote_code=False,dtype=bfloat16",
        tasks=args.tasks.split(","),
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        # Note: 'metadata' isn't a standard simple_evaluate arg in all versions, 
        # but if your previous setup used it to configure tasks, you might need 
        # to pass it to gen_kwargs or task_args depending on lm_eval version.
        # For now, we don't pass it to avoid crashing if unsupported.
    )

    if results:
        print(make_table(results))
        if args.output_path:
            # Inject your custom metadata into the results before saving
            if metadata_dict:
                results["custom_metadata"] = metadata_dict
            
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()