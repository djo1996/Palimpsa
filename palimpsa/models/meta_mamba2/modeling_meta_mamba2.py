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

from palimpsa.models.meta_mamba2.configuration_meta_mamba2 import MetaMamba2Config
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

        self.conv_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.intermediate_size + 2 * config.n_groups * self.state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.mu_states = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            self.num_heads,
            self.head_dim,
            self.state_size,
            device=device,
            dtype=dtype,
        )
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

@dataclass
class MetaMamba2CausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
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
    supports_gradient_checkpointing = True

    def _init_weights(
    self,
    module: nn.Module,
    num_residuals_per_layer: int = 1, # Default as per FLA
    ):
        if isinstance(module, MetaMamba2) and next(module.parameters()).device.type != 'meta':
            with torch.no_grad():
                # 1. A_log (Palimpsa-style uniform log init)
                nn.init.uniform_(module.A_log, a=0, b=16)
                module.A_log.log_()
                module.A_log._no_weight_decay = True
                
                # 2. dt_bias (Discretization math)
                # We create the local tensor and force it in via .data
                dt = torch.exp(
                    nn.init.uniform_(module.dt_bias) * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min)) + math.log(self.config.time_step_min),
                ).clamp(min=1e-4)
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                module.dt_bias.copy_(inv_dt)
                module.dt_bias._no_weight_decay = True

                # 3. b_scale - Metaplasticity scale (Matches Palimpsa log-space init)
                b_scale_min, b_scale_max = 0.1, 10
                b_scale = torch.exp(
                    nn.init.uniform_(module.b_scale) * (math.log(b_scale_max) - math.log(b_scale_min)) + math.log(b_scale_min)
                ).clamp(min=1e-4)
                inv_b_scale = b_scale + torch.log(-torch.expm1(-b_scale))
                module.b_scale.copy_(inv_b_scale)
                module.b_scale._no_weight_decay = True

                # 4. D - Skip connection
                nn.init.ones_(module.D)
                module.D._no_weight_decay = True

                # 5. Metaplasticity
                if hasattr(module, 'b_proj'):
                    std = module.beta_step_rank**-0.5
                    if getattr(self.config, 'finetuning', False):
                        nn.init.uniform_(module.b_scale.data, 0.1, 1.0)
                        nn.init.uniform_(module.b_proj.weight.data, -std, std)
                    else:
                        nn.init.ones_(module.b_scale.data)
                        nn.init.uniform_(module.b_proj.weight.data, -std, std)

        # Standard Layer Initialization
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
                
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        # 5. GPT-2 / Megatron-LM Residual Scaling
        if getattr(self.config, 'rescale_prenorm_residual', False):
            p = None
            if hasattr(module, 'out_proj'):
                p = module.out_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
                
            if p is not None:
                # Re-init then scale by 1/sqrt(N_layers)
                nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
                with torch.no_grad():
                    # N is total layers, use p.data to keep FSDP happy
                    p.data /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)

class MetaMamba2Model(MetaMamba2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MetaMamba2Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm_f = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, cache_params=None, use_cache=None, output_hidden_states=None, return_dict=None, attention_mask=None, **kwargs):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        
        if use_cache and cache_params is None:
            cache_params = MetaMamba2Cache(self.config, inputs_embeds.size(0), device=hidden_states.device, dtype=hidden_states.dtype)

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, cache_params=cache_params, attention_mask=attention_mask, use_cache=use_cache, **kwargs)
        
        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MetaMamba2Output(last_hidden_state=hidden_states, cache_params=cache_params, hidden_states=all_hidden_states)

class MetaMamba2ForCausalLM(MetaMamba2PreTrainedModel, FLAGenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = MetaMamba2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, cache_params=None, output_hidden_states=None, return_dict=None, use_cache=None, attention_mask=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.backbone(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MetaMamba2CausalLMOutput(
            loss=loss, 
            logits=logits, 
            cache_params=outputs.cache_params,
            hidden_states=outputs.hidden_states
        )