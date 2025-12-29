from pathlib import Path
import wandb
import os
from zoology.model import LanguageModel
from zoology.config import LoggerConfig, TrainConfig

class WandbLogger:
    def __init__(self, config: TrainConfig):
        # Check if logger is actually requested
        if config.logger.project_name is None:
            print("No logger specified, skipping...")
            self.no_logger = True
            return
        
        self.no_logger = False
        
        # 1. Handle API Key and Host via Environment Variables (Standard Practice)
        wandb_host = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
        wandb_key = os.environ.get("WANDB_API_KEY")

        if wandb_key:
            wandb.login(key=wandb_key, host=wandb_host)

        # 2. Use the config values instead of hardcoded strings
        self.run = wandb.init(
            entity=config.logger.entity,
            project=config.logger.project_name,
            name=config.run_id,
            config=config.model_dump() # Log the whole config at once
        )

    def log_config(self, config: TrainConfig):
        # Already done in init now, but keeping for compatibility
        pass

    def log_model(self, model: LanguageModel, config: TrainConfig):
        if self.no_logger: return
        
        # Calculate params
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log basics
        metrics = {"num_parameters": params}
        
        # Try to log state size if the model supports it
        try:
            max_seq_len = max([c.input_seq_len for c in config.data.test_configs])
            metrics["state_size"] = model.state_size(sequence_length=max_seq_len)
        except:
            pass
            
        wandb.log(metrics)
        # Avoid wandb.watch(model) in large sweeps, it slows down throughput
    
    def log(self, metrics: dict):
        if not self.no_logger:
            wandb.log(metrics)
    
    def finish(self):
        if not self.no_logger:
            self.run.finish()