from dataclasses import dataclass

@dataclass
class MQARModelConfig:
    # --- Backbone settings (Zoology style) ---
    d_model: int = 128
    n_layers: int = 2
    vocab_size: int = 8192
    max_position_embeddings: int = 512
    layer_norm_epsilon: float = 1e-5
    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    learnable_word_embeddings: bool = True
    pad_vocab_size_multiple: int = 1
    
    # --- Layer specific ---
    layer_name: str = "palimpsa"  # 'palimpsa', 'gla', 'gated_deltanet'
    head_dim: int = 64
    num_heads: int = 2            # d_model (128) / head_dim (64) = 2
    
    # --- Palimpsa specific ---
    expand_v: float = 2
    reduct_k: float = 1
    use_gate: bool = True
    use_short_conv: bool = True
    mode: str = 'chunk'

# === Configurations ===

# 1. Palimpsa (Your settings)
palimpsa_mqar = MQARModelConfig(
    layer_name="palimpsa",
    d_model=128,
    n_layers=2,
    head_dim=64,
    num_heads=2,
    expand_v=2,     # As requested
    reduct_k=1,     # As requested
    use_gate=True,
    mode='chunk'
)

# 2. Gated Linear Attention (GLA)
gla_mqar = MQARModelConfig(
    layer_name="gla",
    d_model=128,
    n_layers=2,
    head_dim=64,
    num_heads=2,
    use_gate=True,
    use_short_conv=True
)

# 3. Gated DeltaNet
gated_deltanet_mqar = MQARModelConfig(
    layer_name="gated_deltanet",
    d_model=128,
    n_layers=2,
    head_dim=64,
    num_heads=2,
    use_gate=True,
    use_short_conv=True
)

# Registry
MQAR_CONFIGS = {
    "palimpsa": palimpsa_mqar,
    "gla": gla_mqar,
    "gated_deltanet": gated_deltanet_mqar
}