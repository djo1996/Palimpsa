# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.palimpsa.configuration_palimpsa import PalimpsaConfig
from fla.models.palimpsa.modeling_palimpsa import PalimpsaBlock, PalimpsaForCausalLM, PalimpsaModel

AutoConfig.register(PalimpsaConfig.model_type, PalimpsaConfig, exist_ok=True)
AutoModel.register(PalimpsaConfig, PalimpsaModel, exist_ok=True)
AutoModelForCausalLM.register(PalimpsaConfig, PalimpsaForCausalLM, exist_ok=True)


__all__ = ['PalimpsaConfig', 'PalimpsaForCausalLM', 'PalimpsaModel', 'PalimpsaBlock']
