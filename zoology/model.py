import math
from functools import partial
import torch
import torch.nn as nn
from zoology.config import ModelConfig

class TokenEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = None
        if config.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)

    def forward(self, input_ids, position_ids=None):
        embeddings = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
            embeddings += self.position_embeddings(position_ids)
        return embeddings
def _init_weights(module, initializer_range=0.02, forgetting_type="normal"):
    from zoology.mixers.gated_delta_net import GatedDeltaNet
    from zoology.mixers.palimpsa import Palimpsa
    from zoology.mixers.meta_mamba2 import MetaMamba2
    
    # Flag to track if we've already handled this specific module's custom logic
    custom_init_applied = False

    # 1. SSM Mixers: Temporal parameters (A and dt)
    if isinstance(module, (GatedDeltaNet, Palimpsa, MetaMamba2)):
        with torch.no_grad():
            n_channels = module.A_log.shape[0]
            
            # Linear Ramp for A
            if forgetting_type == "small":
                vals = torch.linspace(0.01, 0.16, steps=n_channels)
            elif forgetting_type == "very_small":
                vals = torch.linspace(0.001, 0.016, steps=n_channels)
            else:
                vals = torch.linspace(1.0, 16.0, steps=n_channels)
            
            module.A_log.copy_(vals.log())
            module.A_log._no_weight_decay = True

            # Linear Ramp for dt
            dt_ramp = torch.linspace(1.0, 0.0, steps=n_channels)
            dt = torch.exp(
                dt_ramp * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_weight_decay = True
       
            if hasattr(module, 'b_proj') and hasattr(module, 'beta_step_rank'):
                std = module.beta_step_rank ** -0.5 
                nn.init.uniform_(module.b_proj.weight, -std, std)
                if module.b_proj.bias is not None:
                    nn.init.zeros_(module.b_proj.bias)
                
                # If there's a rank projection (hidden_size -> rank)
                if hasattr(module, 'b_rank_proj'):
                    nn.init.normal_(module.b_rank_proj.weight, std=initializer_range)
                
                # We can return here because we've handled the entire mixer's custom parts
                # The .apply() will hit the Linear/Conv sub-modules separately later
                return 

    # This block will handle in_proj, out_proj, etc., but NOT b_proj 
    # because we returned above when we hit the parent mixer module.
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)


class LMBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embeddings = TokenEmbeddings(config)
        
        # Registry for clean block loading
        BLOCK_REGISTRY = {
            "GatedDeltaNetBlock": "zoology.mixers.gated_delta_net",
            "PalimpsaBlock": "zoology.mixers.palimpsa",
            "MetaMamba2Block": "zoology.mixers.meta_mamba2",
        }

        if config.block_type not in BLOCK_REGISTRY:
            raise ValueError(f"Unknown block type: {config.block_type}")

        # Dynamic import to avoid circular dependencies
        module = __import__(BLOCK_REGISTRY[config.block_type], fromlist=[config.block_type])
        block_cls = getattr(module, config.block_type)

        self.layers = nn.ModuleList([
            block_cls(config=config, layer_idx=i) for i in range(config.n_layers)
        ])
        
        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        # Apply standard init
        forgetting_style = getattr(config, "forgetting", "normal")
        print(f"DEBUG: Forgetting style is {forgetting_style}")
        self.apply(partial(
            _init_weights, 
            initializer_range=128 ** -0.5,  
            forgetting_type=forgetting_style
        ))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        # Final residual connection and norm
        hidden_states = self.ln_f((self.drop_f(hidden_states) + residual).to(self.ln_f.weight.dtype))
        return hidden_states

class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # Ensure vocab is padded for efficiency
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)

        self.backbone = LMBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight
        self.to(torch.bfloat16)

    def forward(self, input_ids, position_ids=None): 
        return self.lm_head(self.backbone(input_ids, position_ids=position_ids))