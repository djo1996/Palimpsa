# -*- coding: utf-8 -*-
# Copyright 2024 state-spaces/mamba2 org, HuggingFace Inc. team, and Djohan Bonnet.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.utils.deprecation import deprecate_kwarg

from palimpsa.models.configuration_meta_mamba2 import MetaMamba2Config
from palimpsa.layers.meta_mamba2 import MetaMamba2 
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules.l2warp import l2_warp

logger = logging.get_logger(__name__)

class MetaMamba2Cache(Cache):
    def __init__(
        self,
        config: MetaMamba2Config,
        batch_size: int,
        dtype: torch.dtype = torch.float16,
        device: str | None = None,
    ):
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * config.hidden_size)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.state_size = config.state_size

        # Standard Mamba2/SSM Conv State
        self.conv_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.intermediate_size + 2 * config.n_groups * self.state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        
        # Bayesian States: Mu (SSM state) and I (Importance/Information matrix)
        # Mu: [layers, B, H, D, N]
        self.mu_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=device,
            dtype=dtype,
        )
        # I: [layers, B, H, N, N]
        # Note: This can be memory intensive if state_size is large.
        self.I_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_heads,
            self.state_size,
            self.state_size,
            device=device,
            dtype=dtype,
        )

    def update_conv_state(self, layer_idx: int, new_conv_state: torch.Tensor, cache_init: bool = False):
        if cache_init:
            self.conv_states[layer_idx] = new_conv_state.to(self.conv_states.device)
        else:
            self.conv_states[layer_idx] = self.conv_states[layer_idx].roll(shifts=-1, dims=-1)
            self.conv_states[layer_idx][:, :, -1] = new_conv_state[:, 0, :].to(self.conv_states.device)
        return self.conv_states[layer_idx]

    def update_recurrent_state(self, layer_idx: int, new_state: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if isinstance(new_state, (list, tuple)):
            self.mu_states[layer_idx] = new_state[0].to(self.mu_states.device)
            self.I_states[layer_idx] = new_state[1].to(self.I_states.device)
        else:
            self.mu_states[layer_idx] = new_state.to(self.mu_states.device)
        return (self.mu_states[layer_idx], self.I_states[layer_idx])

@dataclass
class MetaMamba2Output(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    cache_params: MetaMamba2Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None

class MetaMamba2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mixer = MetaMamba2(
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
            state_size=config.state_size,
            expand=config.expand,
            n_groups=config.n_groups,
            conv_kernel=config.conv_kernel,
            use_conv_bias=config.use_conv_bias,
            hidden_act=config.hidden_act,
            rms_norm=config.rms_norm,
            chunk_size=config.chunk_size,
            time_step_rank=config.time_step_rank,
            time_step_limit=config.time_step_limit,
            time_step_min=config.time_step_min,
            time_step_max=config.time_step_max,
            use_bias=config.use_bias,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
            metaplasticity=config.metaplasticity,
            finetuning=config.finetuning,
            beta_step_rank=config.beta_step_rank,
            mode=config.mode,
        )

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states, _, cache_params = self.mixer(
            hidden_states,
            past_key_values=cache_params,
            attention_mask=attention_mask,
            **kwargs
        )
        return residual + hidden_states

class MetaMamba2PreTrainedModel(PreTrainedModel):
    config_class = MetaMamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["MetaMamba2Block"]

    def _init_weights(self, module):
        if isinstance(module, MetaMamba2):
            with torch.no_grad():
                # Init A_log exactly like Mamba2/Palimpsa
                A = torch.arange(1, module.num_heads + 1)
                module.A_log.copy_(torch.log(A))
                nn.init.ones_(module.D)
                
                # Bayesian weight initialization (neutral start)
                if hasattr(module, 'b_proj'):
                    std = module.beta_step_rank**-0.5
                    nn.init.uniform_(module.b_proj.weight, -std, std)
                    if module.finetuning:
                        nn.init.uniform_(module.b_scale, 0.1, 1.0)
                    else:
                        nn.init.ones_(module.b_scale)
        
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

class MetaMamba2Model(MetaMamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MetaMamba2Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm_f = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_init()

    def forward(self, input_ids=None, cache_params=None, use_cache=None, attention_mask=None, **kwargs):
        hidden_states = self.embeddings(input_ids)
        
        if use_cache and cache_params is None:
            cache_params = MetaMamba2Cache(self.config, input_ids.size(0), device=hidden_states.device, dtype=hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cache_params=cache_params, attention_mask=attention_mask, use_cache=use_cache, **kwargs)
        
        return MetaMamba2Output(last_hidden_state=self.norm_f(hidden_states), cache_params=cache_params)

class MetaMamba2ForCausalLM(MetaMamba2PreTrainedModel, FLAGenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = MetaMamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        outputs = self.backbone(input_ids, **kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MetaMamba2Output(loss=loss, logits=logits, cache_params=outputs.cache_params)