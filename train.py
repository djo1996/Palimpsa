import sys
import os
import shutil
import torch
from transformers import AutoTokenizer
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
    Creates a reproducible artifact in the experiment folder:
    1. Saves Tokenizer
    2. Copies the raw config.json (foolproof)
    3. Copies the source code (Palimpsa library) to trace exact logic
    """
    # Only the main process (rank 0) should do IO operations
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    dump_folder = config.job.dump_folder
    os.makedirs(dump_folder, exist_ok=True)
    
    print(f"📦 [Snapshot] Creating experiment artifact in: {dump_folder}")

    # ---------------------------------------------------------
    # 1. Save Tokenizer
    # ---------------------------------------------------------
    tokenizer_path = config.model.tokenizer_path
    try:
        print(f"   ├── Saving tokenizer from '{tokenizer_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        tokenizer.save_pretrained(dump_folder)
    except Exception as e:
        print(f"   ⚠️ Failed to save tokenizer: {e}")

    # ---------------------------------------------------------
    # 2. Force Copy Config (The Foolproof Way)
    # ---------------------------------------------------------
    # We don't rely on AutoConfig.save_pretrained() because it can fail with custom models.
    # We simply copy the input JSON file to 'config.json'.
    source_config_path = config.model.config
    dest_config_path = os.path.join(dump_folder, "config.json")
    
    try:
        print(f"   ├── Copying config raw: {source_config_path} -> {dest_config_path}")
        shutil.copy(source_config_path, dest_config_path)
    except Exception as e:
        print(f"   ⚠️ Failed to copy config file: {e}")

    # ---------------------------------------------------------
    # 3. Snapshot Source Code
    # ---------------------------------------------------------
    # We copy the 'palimpsa' library folder into 'exp/.../src/palimpsa'
    # This ensures you have the exact layers/kernels used for this run.
    
    # Assuming train.py is in Palimpsa/train.py, the lib is in Palimpsa/palimpsa
    repo_root = os.path.dirname(os.path.abspath(__file__))
    source_lib_path = os.path.join(repo_root, "palimpsa")
    dest_src_path = os.path.join(dump_folder, "src", "palimpsa")

    if os.path.exists(source_lib_path):
        print(f"   ├── Snapshotting source code to: {dest_src_path}")
        
        # Remove previous snapshot if it exists (e.g. restarting a run)
        if os.path.exists(dest_src_path):
            shutil.rmtree(dest_src_path)

        try:
            shutil.copytree(
                source_lib_path, 
                dest_src_path,
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '*.egg-info')
            )
        except Exception as e:
             print(f"   ⚠️ Failed to snapshot source code: {e}")
    else:
        print(f"   ⚠️ Could not find source library at {source_lib_path}")

    print("   ✅ Snapshot complete.")

if __name__ == "__main__":
    init_logger()
    
    config = JobConfig()
    config.parse_args()
    
    # --- Snapshot before training starts ---
    snapshot_experiment(config)
    
    main(config)
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()