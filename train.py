# Palimpsa/train.py
import sys
import os
import torch
from torchtitan.tools.logging import init_logger

# -----------------------------------------------------------------------------
# 1. Register Palimpsa (The Plugin)
# -----------------------------------------------------------------------------
# This imports your integration.py which registers the model specs.
# This is the "Bridge" that allows Flame to see your model without hacking source code.
import palimpsa.integration 
print("⚡ [Palimpsa] Plugin loaded and models registered.")

# -----------------------------------------------------------------------------
# 2. Launch Flame (The Engine)
# -----------------------------------------------------------------------------
# We import flame's main logic directly. 
# Since we installed flame via pip ('pip install -e .'), this works as a library call!
from flame.main import main, JobConfig

if __name__ == "__main__":
    # Initialize the standard Flame logger
    init_logger()
    
    # Load the configuration (CLI args + YAML file)
    config = JobConfig()
    config.parse_args()
    
    # 3. Run the Training Engine
    main(config)
    
    # Cleanup distributed processes if necessary
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()