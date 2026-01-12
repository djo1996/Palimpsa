import sys
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoConfig
from torchtitan.tools.logging import init_logger

# =============================================================================
# 1. Path Hack for Flame
# =============================================================================
current_dir = os.getcwd()
flame_root = os.path.join(current_dir, "flame")
if os.path.exists(flame_root):
    sys.path.append(flame_root)
else:
    print(f"⚠️ Warning: Could not find flame root at {flame_root}")

# =============================================================================
# 2. Register Palimpsa (The Plugin)
# =============================================================================
import palimpsa.integration 
print("⚡ [Palimpsa] Plugin loaded and models registered.")

# =============================================================================
# 3. Import Flame Engine
# =============================================================================
from flame.train import main
from flame.config_manager import JobConfig

def snapshot_experiment(config):
    """
    Saves the tokenizer and model config to the experiment folder (dump_folder).
    This ensures the 'exp' folder is a self-contained artifact.
    """
    # Only the main process (rank 0) should perform IO
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    dump_folder = config.job.dump_folder
    os.makedirs(dump_folder, exist_ok=True)
    
    print(f"📦 [Snapshot] Preparing experiment artifact in: {dump_folder}")

    # 1. Save Tokenizer
    tokenizer_path = config.model.tokenizer_path
    try:
        print(f"   ├── Saving tokenizer from '{tokenizer_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tokenizer.save_pretrained(dump_folder)
    except Exception as e:
        print(f"   ⚠️ Failed to save tokenizer: {e}")

    # 2. Copy/Save Config
    # We load the config to ensure validity, then save the standard 'config.json'
    model_config_path = config.model.config
    try:
        print(f"   ├── Saving config from '{model_config_path}'...")
        hf_config = AutoConfig.from_pretrained(model_config_path, trust_remote_code=True)
        hf_config.save_pretrained(dump_folder)
        
        # Backup the exact original JSON just in case
        shutil.copy(model_config_path, os.path.join(dump_folder, "original_training_config.json"))
    except Exception as e:
        print(f"   ⚠️ Failed to save config: {e}")
    
    print("   ✅ Snapshot complete.")

if __name__ == "__main__":
    init_logger()
    
    # Parse args
    config = JobConfig()
    config.parse_args()
    
    # --- NEW: Snapshot the configuration before training starts ---
    snapshot_experiment(config)
    
    # Launch Flame
    main(config)
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()