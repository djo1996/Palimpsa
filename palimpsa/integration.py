# Palimpsa/palimpsa/integration.py
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from flame.models.parallelize_fla import parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.loss import build_cross_entropy_loss
from flame.data import build_dataloader
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

# Import YOUR custom model
# Ensure your model file has PalimpsaForCausalLM and PalimpsaConfig
from .models.palimpsa import PalimpsaForCausalLM, PalimpsaConfig

# =============================================================================
# 1. HuggingFace AutoClass Registration
# =============================================================================
# This patches AutoConfig so Flame can load your config without source hacks.
# Flame calls AutoConfig.from_pretrained(), which will now find "palimpsa".
try:
    AutoConfig.register("palimpsa", PalimpsaConfig)
    AutoModelForCausalLM.register(PalimpsaConfig, PalimpsaForCausalLM)
except ValueError:
    pass # Already registered

# =============================================================================
# 2. Flame Registry
# =============================================================================
def build_tokenizer(job_config):
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)

register_train_spec(
    TrainSpec(
        name="palimpsa",  # Matches YAML model.name
        cls=PalimpsaForCausalLM,
        config=PalimpsaConfig,
        parallelize_fn=parallelize_fla,
        pipelining_fn=pipeline_fla,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)