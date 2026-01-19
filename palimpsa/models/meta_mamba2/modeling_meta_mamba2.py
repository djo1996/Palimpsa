# -*- coding: utf-8 -*-
# Copyright 2024 state-spaces/mamba2 org, HuggingFace Inc. team, and Djohan Bonnet.


from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from palimpsa.models.meta_mamba2.configuration_meta_mamba2 import MetaMamba2Config
from palimpsa.layers.meta_mamba2 import MetaMamba2 
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules.l2warp import l2_warp

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)



class MetaMamba2Block(GradientCheckpointingLayer):
    def __init__(self, config: MetaMamba2Config, layer_idx: int):
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
            init_diagnosis=getattr(config, "init_diagnosis", False),
            eval_diagnosis=getattr(config, "eval_diagnosis", False),
        )
        self.residual_in_fp32 = getattr(config, 'residual_in_fp32', False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states, attentions, past_key_values = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        hidden_states = residual + hidden_states
        if self.residual_in_fp32:
            hidden_states = hidden_states.to(dtype=self.norm.weight.dtype)
        outputs = (hidden_states, attentions, past_key_values)

        return outputs

class MetaMamba2PreTrainedModel(PreTrainedModel):
    config_class = MetaMamba2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["MetaMamba2Block"]
    supports_gradient_checkpointing = True
    _supports_cache_class = True

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
                b_scale_min, b_scale_max = 0.1, 1
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

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[dict],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`MetaMamba2Model` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds
        
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)


        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            if output_attentions:
                all_attns += (attentions,)
        
        hidden_states = self.norm_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values, hidden_states=all_hidden_states, attentions=all_attns,)

class MetaMamba2ForCausalLM(MetaMamba2PreTrainedModel, FLAGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.backbone = MetaMamba2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.embeddings

    def set_input_embeddings(self, value):
        self.backbone.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.backbone = decoder

    def get_decoder(self):
        return self.backbone

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead."
                )
            else:
                raise exception

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[dict],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs[0]

        loss, logits = None, None
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )