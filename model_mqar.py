import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Imports from FLA ---
# We use the primitives (Norms, MLP) from FLA to build the Blocks
try:
    from fla.modules import RMSNorm, GatedMLP
except ImportError:
    print("⚠️ Warning: fla not found. Please install flash-linear-attention.")

# --- Robust Layer Imports ---
def get_layer_class(layer_name):
    """Dynamically import the attention layer based on name."""
    if layer_name.lower() == 'palimpsa':
        try:
            # Try local package first, then FLA
            from palimpsa.models.palimpsa.modeling_palimpsa import Palimpsa
            return Palimpsa
        except ImportError:
            try:
                from fla.layers.palimpsa import Palimpsa
                return Palimpsa
            except ImportError:
                raise ImportError("Could not import Palimpsa layer from 'palimpsa' or 'fla'.")
                
    elif layer_name.lower() == 'gla':
        from fla.layers.gla import GatedLinearAttention
        return GatedLinearAttention
        
    elif layer_name.lower() == 'gated_deltanet':
        # FLA export name usually 'GatedDeltaNet' or 'SimpleGatedDeltaNet'
        from fla.layers.delta_net import GatedDeltaNet
        return GatedDeltaNet
    
    else:
        raise ValueError(f"Unknown layer type: {layer_name}")

# =============================================================================
# 1. Generic Block Wrapper (Adapts FLA Layers to Zoology Loop)
# =============================================================================
class FLABlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.d_model
        
        # 1. Norms
        self.attn_norm = RMSNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp_norm = RMSNorm(self.hidden_size, eps=config.layer_norm_epsilon)

        # 2. Attention Layer (Dynamic Loading)
        # We pass 'layer_name' in the config, defaulting to Palimpsa
        layer_cls = get_layer_class(getattr(config, 'layer_name', 'palimpsa'))
        
        self.attn = layer_cls(
            hidden_size=config.d_model,
            layer_idx=layer_idx,
            mode='chunk', # Force chunk mode for efficiency
            # FLA/Palimpsa standard args
            head_dim=getattr(config, 'head_dim', 64),
            num_heads=getattr(config, 'num_heads', 4),
            use_short_conv=True
        )

        # 3. MLP
        self.mlp = GatedMLP(
            hidden_size=config.d_model,
            hidden_ratio=4,
            # FIXED: Removed 'act_fn' argument as it caused the crash.
            # FLA GatedMLP defaults to Swish/SiLU automatically.
        )

    def forward(self, hidden_states, residual):
        # Zoology expects: (hidden, residual) -> (hidden, residual)
        
        # --- Attention Branch ---
        normed = self.attn_norm(hidden_states)
        
        # FLA layers return (output, cache, ...) - we just want output
        # Some layers might return a single tensor, others a tuple. Handle both.
        attn_out = self.attn(normed)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
            
        if residual is None:
            residual = hidden_states
            hidden_states = attn_out
        else:
            hidden_states = residual + attn_out
            residual = hidden_states

        # --- MLP Branch ---
        normed = self.mlp_norm(hidden_states)
        mlp_out = self.mlp(normed)
        
        hidden_states = hidden_states + mlp_out
        residual = hidden_states

        return hidden_states, residual

# =============================================================================
# 2. Zoology Backbone (Preserved)
# =============================================================================

class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        learnable: bool = True,
        device='cuda',
        dtype='torch.bfloat16',
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size, word_embed_proj_dim, padding_idx=padding_idx
            )
            self.project_in = nn.Linear(word_embed_proj_dim, embed_dim, bias=False)
            
        if not learnable:
            self.word_embeddings.weight.requires_grad = False

        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim
            )

    def forward(self, input_ids, position_ids=None):
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
            
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=embeddings.device
                )
            # Support broadcasting if necessary, though Zoology usually strict
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

def _init_weights(
        module,
        n_layers,
        block_type,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

class LMBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = TokenEmbeddings(
            config.d_model, 
            config.vocab_size, 
            config.max_position_embeddings,
            learnable=config.learnable_word_embeddings
        )
        
        # We use the generic wrapper that checks 'config.layer_name'
        block_cls = FLABlock

        self.layers = nn.ModuleList(
            [
                block_cls(config=config, layer_idx=i)
                for i in range(config.n_layers)
            ]
        )
        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.apply(partial(_init_weights, n_layers=config.n_layers, block_type="FLABlock"))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
            
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )

        self.backbone = LMBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights 
        self.apply(partial(_init_weights, n_layers=config.n_layers, block_type="FLABlock"))

        # tie weights
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight
        self.to(torch.bfloat16)

    def forward(
        self, input_ids, position_ids=None, state=None, labels=None
    ): 
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        logits = self.lm_head(hidden_states)
        
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Return HF-like output
            from collections import namedtuple
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "loss"])
            return CausalLMOutput(logits=logits, loss=loss)

        return logits

    def state_size(self, sequence_length: int):
        return "dont care"