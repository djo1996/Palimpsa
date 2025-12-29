import os
import wandb
from zoology.model import LanguageModel
from zoology.config import TrainConfig

class WandbLogger:
    def __init__(self, config: TrainConfig):
        # 1. Skip if no project is defined
        if config.logger.project_name is None:
            print("No W&B project specified, skipping logging...")
            self.no_logger = True
            return
        
        self.no_logger = False

        # 2. Setup Host (Default to public W&B if not set in env)
        # On your cluster, you should: export WANDB_BASE_URL="https://wandb.fz-juelich.de"
        wandb_host = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
        wandb_key = os.environ.get("WANDB_API_KEY")

        if wandb_key:
            # Only login if a key is provided; otherwise assume local machine is already 'wandb login'ed
            wandb.login(key=wandb_key, host=wandb_host)

        # 3. Initialize Run using the Config values
        self.run = wandb.init(
            entity=config.logger.entity,
            project=config.logger.project_name,
            name=config.run_id,
            config=config.model_dump()
        )

    def log_model(self, model: LanguageModel, config: TrainConfig):
        if self.no_logger:
            return
        
        # Log parameter count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics = {"num_parameters": params}
        
        # Try to log state size
        try:
            max_seq_len = max([c.input_seq_len for c in config.data.test_configs])
            metrics["state_size"] = model.state_size(sequence_length=max_seq_len)
        except Exception:
            pass
            
        self.run.log(metrics)

    def log(self, metrics: dict):
        if not self.no_logger:
            self.run.log(metrics)
    
    def finish(self):
        if not self.no_logger:
            self.run.finish()