# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_meta_mamba2 import MetaMamba2Config
from .modeling_meta_mamba2 import MetaMamba2Block, MetaMamba2ForCausalLM, MetaMamba2Model

# Registering the Meta-Mamba2 architecture into Transformers
AutoConfig.register(MetaMamba2Config.model_type, MetaMamba2Config, exist_ok=True)
AutoModel.register(MetaMamba2Config, MetaMamba2Model, exist_ok=True)
AutoModelForCausalLM.register(MetaMamba2Config, MetaMamba2ForCausalLM, exist_ok=True)

__all__ = ['MetaMamba2Config', 'MetaMamba2ForCausalLM', 'MetaMamba2Model', 'MetaMamba2Block']