from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from flame.models.parallelize_fla import parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.loss import build_cross_entropy_loss
from flame.data import build_dataloader
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

# --- FIX: Use absolute import here ---
from palimpsa.models.palimpsa import PalimpsaForCausalLM, PalimpsaConfig

# 1. AutoClass Registration (For HuggingFace)
try:
    AutoConfig.register("palimpsa", PalimpsaConfig)
    AutoModelForCausalLM.register(PalimpsaConfig, PalimpsaForCausalLM)
except ValueError:
    pass 

# 2. Flame Registry (For Training Engine)
def build_tokenizer(job_config):
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)

register_train_spec(
    TrainSpec(
        name="palimpsa",
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