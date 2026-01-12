import argparse
import os
import io
import torch
from datetime import timedelta
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- Register Palimpsa Models ---
from palimpsa.models.palimpsa import PalimpsaConfig, PalimpsaForCausalLM
from palimpsa.models.meta_mamba2 import MetaMamba2Config, MetaMamba2ForCausalLM

AutoConfig.register("palimpsa", PalimpsaConfig)
AutoModelForCausalLM.register(PalimpsaConfig, PalimpsaForCausalLM)
AutoConfig.register("meta_mamba2", MetaMamba2Config)
AutoModelForCausalLM.register(MetaMamba2Config, MetaMamba2ForCausalLM)

def convert_dcp_to_hf(exp_dir, step, output_dir=None):
    """
    Converts a Distributed Checkpoint (DCP) to Hugging Face format.
    Assumes 'config.json' and tokenizer files exist in 'exp_dir' (saved by train.py).
    """
    dcp_path = os.path.join(exp_dir, "checkpoint", f"step-{step}")
    
    # Default output: exp/model_name/hf_model_step_X
    if output_dir is None:
        output_dir = os.path.join(exp_dir, f"hf_model_step_{step}")
    
    print(f"📂 Experiment Dir: {exp_dir}")
    print(f"🔄 Converting Step: {step}")
    print(f"🎯 Output Dir:     {output_dir}")

    if not os.path.exists(dcp_path):
        raise FileNotFoundError(f"Checkpoint not found at: {dcp_path}")

    # 1. Load Artifacts from Experiment Root
    print("   ├── Loading Config & Tokenizer from experiment root...")
    try:
        config = AutoConfig.from_pretrained(exp_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(exp_dir, trust_remote_code=True)
    except OSError:
        print("   ❌ Error: config.json or tokenizer not found in experiment dir.")
        print("      Did you run the new train.py? If not, copy them there manually.")
        return

    # 2. Consolidate DCP to Temp File
    os.makedirs(output_dir, exist_ok=True)
    temp_pt_path = os.path.join(output_dir, "temp_weights.pt")
    
    print("   ├── Consolidating Distributed Checkpoint (DCP)...")
    dcp_to_torch_save(dcp_path, temp_pt_path)

    # 3. Load and Save as HF
    print("   ├── Loading state dict and saving HF model...")
    torch.serialization.add_safe_globals([timedelta, io.BytesIO])
    
    # Load weights
    state_dict = torch.load(temp_pt_path, map_location="cpu", weights_only=False)['model']
    
    # Init model & Load
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Cleanup
    if os.path.exists(temp_pt_path):
        os.remove(temp_pt_path)
    
    print("   ✅ Conversion Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Path to experiment folder (e.g. exp/palimpsa-170M)")
    parser.add_argument("--step", type=int, required=True, help="Step number to convert (e.g. 3000)")
    parser.add_argument("--out", type=str, default=None, help="Optional custom output path")
    args = parser.parse_args()

    convert_dcp_to_hf(args.exp, args.step, args.out)