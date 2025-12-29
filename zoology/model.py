import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.ops import StochasticDepth

from zoology.config import ModelConfig


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
        """
        GPT-2 Learnable Token and Position Embeddings.
        If max_position_embeddings <= 0, there's no position embeddings
        Wwe embed to word_embe_proj_dim dimension then project up to embed_dim
        """
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
                vocab_size,
                word_embed_proj_dim,
                padding_idx=padding_idx,
            )
            self.project_in = nn.Linear(
                word_embed_proj_dim, embed_dim, bias=False
            )
        if not learnable:
            self.word_embeddings.weight.requires_grad = False

        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim
            )

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=self.device
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings

def _init_weights(
        module,
        n_layers,
        block_type,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
        
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

       
# def _init_weights(
#     module,
#     n_layers,
#     block_type,
#     initializer_range=0.02,
#     rescale_prenorm_residual=True,
#     n_residuals_per_layer=1,  # Change to 2 if we have MLP
# ):
  
   
    
#     if rescale_prenorm_residual:
#         # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
#         #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
#         #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
#         #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
#         #
#         # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
#         for name, p in module.named_parameters():
#             if name in ["out_proj.weight", "fc2.weight"]:
#                 # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
#                 # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
#                 # We need to reinit p since this code could be called multiple times
#                 # Having just p *= scale would repeatedly scale it down
#                 nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#                 with torch.no_grad():
#                     p /= math.sqrt(n_residuals_per_layer * n_layers)





class LMBackbone(nn.Module):
    def __init__(self, config: ModelConfig):

        super().__init__()
        self.embeddings = TokenEmbeddings(
            config.d_model, 
            config.vocab_size, 
            config.max_position_embeddings,
            learnable=config.learnable_word_embeddings
        )
        
        print('config.block_type' , config.block_type)
        if config.block_type == 'BMAHeadsBlock':
            from zoology.mixers.bma_heads import BMAHeadsBlock
            print('the config_block_type is ', config.block_type)
            print("loading bayesian metaplsatic attention block")
            block_cls = BMAHeadsBlock
        elif config.block_type == 'GatedDeltaNetBlock':
            from zoology.mixers.gated_delta_net import GatedDeltaNetBlock
            block_cls = GatedDeltaNetBlock
            print("loading GatedDeltaNet block")
        elif config.block_type == 'PalimpsaBlock':
            from zoology.mixers.palimpsa import PalimpsaBlock
            block_cls = PalimpsaBlock
            print("loading Palimpsa block")

        self.layers = nn.ModuleList(
            [
                block_cls(config=config, layer_idx=i)
                for i in range(config.n_layers)
            ]
        )
        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.apply(partial(_init_weights, n_layers=config.n_layers, block_type=config.block_type))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(
            input_ids,
            position_ids=position_ids,
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        return hidden_states


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )

        self.backbone = LMBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights, n_layers=config.n_layers, block_type=config.block_type))

        # tie weights
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight
        self.to(torch.bfloat16)

    def forward(
        self, input_ids, position_ids=None, state=None
    ): 
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        return self.lm_head(hidden_states)
    def state_size(self,sequence_length: int):
        return("dont care")
  