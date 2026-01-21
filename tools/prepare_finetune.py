# -*- coding: utf-8 -*-
import argparse
import os
import io
import torch
from datetime import timedelta
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
import torch.distributed.checkpoint as DCP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- Register custom architectures for the Lab ---
# These imports ensure the Python classes are available in the local namespace.
from palimpsa.models.palimpsa import PalimpsaConfig, PalimpsaForCausalLM
from palimpsa.models.meta_mamba2 import MetaMamba2Config, MetaMamba2ForCausalLM

# Mapping the 'model_type' from your config.json to the specific classes.
# This allows AutoModelForCausalLM.from_config(config) to instantiate your model.
AutoConfig.register("palimpsa", PalimpsaConfig)
AutoModelForCausalLM.register(PalimpsaConfig, PalimpsaForCausalLM)

AutoConfig.register("meta_mamba2", MetaMamba2Config)
AutoModelForCausalLM.register(MetaMamba2Config, MetaMamba2ForCausalLM)

@torch.inference_mode()
def perform_surgery(args):
    """
    Safely transforms a non-metaplastic base model into a finetune-ready Palimpsa model.
    """
    print(f"🚀 Starting Surgery: {args.src_exp} -> {args.dst_dir}")
    
    # 1. Load the new target configuration and tokenizer.
    # The config defines the new 'metaplasticity' status and 'beta_step_rank'.
    config = AutoConfig.from_pretrained(args.new_config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # 2. Create the exact Flame-compatible structure: dst_dir/checkpoint/step-0
    # This structure is required for --checkpoint.load_step 0 to work.
    checkpoint_step_dir = os.path.join(args.dst_dir, "checkpoint", "step-0")
    os.makedirs(checkpoint_step_dir, exist_ok=True)
    temp_pt = os.path.join(args.dst_dir, "temp_surgery.pt")
    
    # 3. Consolidate source DCP (which may be sharded) to a single temporary file.
    # This allows us to load and manipulate the weights on the CPU easily.
    src_dcp = os.path.join(args.src_exp, "checkpoint", f"step-{args.step}")
    print(f"📦 Consolidating source checkpoint from step {args.step}...")
    dcp_to_torch_save(src_dcp, temp_pt)
    
    # 4. Load weights and purge outdated/mismatched plasticity keys.
    torch.serialization.add_safe_globals([timedelta, io.BytesIO])
    full_sd = torch.load(temp_pt, map_location='cpu', weights_only=False)
    state_dict = full_sd['model'] if 'model' in full_sd else full_sd
    
    # We PURGE the plasticity parameters from the old model.
    # Because they are missing, the NEW model will use its fresh, correctly-shaped init.
    keys_to_purge = [k for k in state_dict.keys() if any(sub in k for sub in ["b_proj", "b_rank_proj", "b_scale", "Ip_log", "bs_proj"])]
    for k in keys_to_purge:
        del state_dict[k]
    print(f"✂️ Purged {len(keys_to_purge)} outdated keys. Ready for fresh meta-init.")

    # 5. Initialize new model and load the preserved backbone weights.
    # 'strict=False' is critical because we intentionally removed the plasticity parameters.
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict, strict=False)
    
    # 6. Save as a sharded DCP with the required .metadata file.
    # Flame's loading mechanism will find this pancake-flat file and initialize step 0.
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint_step_dir)
    DCP.save(model.state_dict(), storage_writer=storage_writer)
    
    # 7. Finalize the new experiment folder with standard HF files.
    model.save_pretrained(os.path.join(args.dst_dir, "hf_model"))
    tokenizer.save_pretrained(args.dst_dir)
    config.save_pretrained(args.dst_dir)
    
    # Cleanup to save disk space.
    if os.path.exists(temp_pt):
        os.remove(temp_pt)
    print(f"🏁 Finished. Ready for torchrun at {args.dst_dir} with --checkpoint.load_step 0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_exp", type=str, required=True, help="Path to original experiment")
    parser.add_argument("--step", type=int, required=True, help="Source checkpoint step")
    parser.add_argument("--new_config", type=str, required=True, help="JSON for the finetune config")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="HF Tokenizer name or path")
    parser.add_argument("--dst_dir", type=str, required=True, help="Fresh destination experiment folder")
    args = parser.parse_args()
    perform_surgery(args)