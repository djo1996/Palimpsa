# -*- coding: utf-8 -*-

from __future__ import annotations
from torch import Tensor

import math
import warnings
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
import torch.nn.functional as F 

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack
    from fla.models.utils import Cache

from palimpsa.layers.palimpsa import Palimpsa

class PalimpsaBlock(nn.Module):
    def __init__(
        self, config, fused_add_norm=True, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = Palimpsa(d_model, **factory_kwargs, **config.sequence_mixer.kwargs)
        self.norm = RMSNorm(d_model, eps=norm_epsilon)
        
    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states, _, _ = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual  

