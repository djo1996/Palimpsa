````python
# -*- coding: utf-8 -*-

import math
import warnings
from typing import Optional, Union, Tuple
from transformers.configuration_utils import PretrainedConfig

class MetaMamba2Config(PretrainedConfig):
    model_type = "metamamba2"

    def __init__(
        self,
        head_dim: int = 64,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        state_size: int = 128,
        num_hidden_layers: int = 48,
        norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        expand: int = 2,
        conv_kernel: int = 4,
        n_groups: int = 1,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        residual_in_fp32: bool = True,
        # Bayesian Metaplasticity bits
        metaplasticity: bool = True,
        beta_step_rank: Union[str, int] = "auto",
        finetuning: bool = False,
        # Discretization
        time_step_rank: Union[str, int] = "auto",
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 1e-4,
        time_step_limit: Tuple[float, float] = (0.0, float("inf")),
        # FLA Runtime / Fusing logic
        rescale_prenorm_residual: bool = True,
        use_cache: bool = True,
        rms_norm: bool = True,
        chunk_size: int = 256,
        mode: str = 'chunk',
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.residual_in_fp32 = residual_in_fp32
        
        # Bayesian / Metaplasticity config
        self.metaplasticity = metaplasticity
        self.finetuning = finetuning
        self.mode = mode
        if beta_step_rank == "auto":
            self.beta_step_rank = math.ceil(self.hidden_size / 16)
        else:
            self.beta_step_rank = beta_step_rank

        # Discretization mapping
        if time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.hidden_size / 16)
        else:
            self.time_step_rank = time_step_rank
        
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.time_step_limit = time_step_limit
        
        # Derived Mamba2 Head logic
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.num_heads = int(self.expand * self.hidden_size / self.head_dim)
        
        # Optimization flags
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.use_cache = use_cache
        self.rms_norm = rms_norm
        self.chunk_size = chunk_size
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.tie_word_embeddings = tie_word_embeddings

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.")

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )