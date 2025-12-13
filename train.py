import sys
import os
import torch
from torchtitan.tools.logging import init_logger

# 1. Register Palimpsa (The Plugin)
import palimpsa.integration 
print("⚡ [Palimpsa] Plugin loaded and models registered.")

# 2. Launch Flame (The Engine)
from flame.train import main
from flame.config_manager import JobConfig

if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.parse_args()
    main(config)
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()