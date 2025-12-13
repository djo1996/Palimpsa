import sys
import os
import torch
from torchtitan.tools.logging import init_logger

# =============================================================================
# 1. Path Hack for Flame
# =============================================================================
# Flame imports 'custom_models', which sits in the root of the flame repo.
# We must add that directory to sys.path so Python can find it.
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
# Now this import will succeed because 'custom_models' is visible
from flame.train import main
from flame.config_manager import JobConfig

if __name__ == "__main__":
    init_logger()
    
    config = JobConfig()
    config.parse_args()
    
    main(config)
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()