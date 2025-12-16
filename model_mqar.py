import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple

# --- FLA Imports ---
try:
    from fla.modules import RMSNorm, GatedMLP
except ImportError:
    print("⚠️ Warning: fla not found. Please install flash-linear-attention.")

# --- Dynamic Layer Loader ---
def get_mixer_class(layer_name: str):
    """
    Dynamically imports the mixing layer (Attention/SSM) from FLA or local packages.
    """
    name = layer_name.lower()
    
    # 1. Palimpsa
    if name == 'palimpsa':
        try:
            from fla.layers.palimpsa import Palimpsa
            return Palimpsa
        except ImportError:
            # Fallback for local dev
            from palimpsa.models.palimpsa.modeling_palimpsa import Palimpsa
            return Palimpsa

    # 2. Gated Linear Attention (GLA)
    elif name == 'gla':
        from fla.layers.gla import GatedLinearAttention
        return GatedLinearAttention

    # 3. Gated DeltaNet
    elif name == 'gated_deltanet':
        from fla.layers.gated_deltanet import GatedDeltaNet
        return GatedDeltaNet
    
    # 4. Standard Attention (Optional fallback)
    elif name == 'attention':
        from fla.layers.attn import Attention
        return Attention

    else:
        raise ValueError(f"Unknown layer type: {layer_name}")


class UnifiedBlock(nn.Module):
    """
    A generic Transformer-style block that composes:
    Norm -> Mixer (Attn/SSM) -> Residual -> Norm -> MLP -> Residual
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        
        # 1. Layer Norms (using RMSNorm from FLA for speed)
        self.attn_norm = RMSNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp_norm = RMSNorm(self.hidden_size, eps=config.layer_norm_epsilon)

        # 2. The Mixer (The "Attention" part)
        mixer_cls = get_mixer_class(getattr(config, 'layer_name', 'palimpsa'))
        
        # We filter args to ensure we don't pass unused ones if the mixer is simple
        self.mixer = mixer_cls(
            hidden_size=config.d_model,
            layer_idx=layer_idx,
            mode=getattr(config, 'mode', 'chunk'),  # Default to efficient chunk mode
            head_dim=getattr(config, 'head_dim', 64),
            num_heads=getattr(config, 'num_heads', 4),
            # Add specific args here if needed (e.g. use_short_conv for FLA)
            use_short_conv=True 
        )

        # 3. The MLP (The "Thinking" part)
        # FLA layers are just mixers, so we MUST explicitly add the MLP here.
        self.mlp = GatedMLP(
            hidden_size=config.d_model,
            hidden_ratio=getattr(config, 'mlp_expansion_factor', 4),
            # FLA's GatedMLP uses Swish/SiLU by default
        )

    def forward(self, hidden_states, residual=None):
        # 1. Mixer Branch
        normed = self.attn_norm(hidden_states)
        
        # Handle FLA layers returning tuples (output, cache)
        mixer_out = self.mixer(normed)
        if isinstance(mixer_out, tuple):
            mixer_out = mixer_out[0]
            
        if residual is None:
            residual = hidden_states
            hidden_states = mixer_out
        else:
            hidden_states = residual + mixer_out
            residual = hidden_states

        # 2. MLP Branch
        normed = self.mlp_norm(hidden_states)
        mlp_out = self.mlp(normed)
        
        # Standard Pre-Norm Residual connection
        hidden_states = hidden_states + mlp_out
        residual = hidden_states

        return hidden_states, residual


class TokenEmbeddings(nn.Module):
    """Standard Learnable Token + Position Embeddings"""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=None
        )
        
        if not getattr(config, 'learnable_word_embeddings', True):
            self.word_embeddings.weight.requires_grad = False

        self.max_position_embeddings = config.max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.d_model
            )

    def forward(self, input_ids, position_ids=None):
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
            
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=embeddings.device
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
            
        return embeddings


class LMBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = TokenEmbeddings(config)
        self.layers = nn.ModuleList([
            UnifiedBlock(config=config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # Final Norm
        self.ln_f = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.drop_f = nn.Dropout(getattr(config, 'resid_dropout', 0.0))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids)
        residual = None
        
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
            
        # Final Residual + Norm
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual)
        
        return hidden_states


def _init_weights(module, n_layers, initializer_range=0.02):
    """Universal Weight Initialization"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Pad vocab if necessary for tensor core efficiency
        pad_multiple = getattr(config, 'pad_vocab_size_multiple', 8)
        if config.vocab_size % pad_multiple != 0:
            config.vocab_size += pad_multiple - (config.vocab_size % pad_multiple)

        self.backbone = LMBackbone(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (Optional, but standard for small LMs)
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

        # Init
        self.apply(partial(_init_weights, n_layers=config.n_layers))
        self.to(torch.bfloat16) # Default to bf16

    def forward(self, input_ids, position_ids=None, labels=None, state=None): 
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        logits = self.lm_head(hidden_states)
        
        if labels is not None:
            # Flatten for CrossEntropy: [Batch * Seq, Vocab]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Use a simple namespace/dict to return both
            from collections import namedtuple
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "loss"])
            return CausalLMOutput(logits=logits, loss=loss)

        return logits

    def state_size(self, sequence_length: int = 0):
        # Used by Zoology logger, can remain dummy
        return 0