import uuid
import os
import numpy as np
import torch
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig
import datetime

# =================================================================
# 1. Environment & Seed Setup
# =================================================================
SWEEP_SEED = 3

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SWEEP_SEED)

sweep_id = uuid.uuid4().hex[:6]
VOCAB_SIZE = 8_192

# Use a local cache directory relative to the project root
# This avoids "Could not create directory" permission errors on the cluster
CACHE_DIR = os.path.join(os.getcwd(), "cache/mqar")

configs = []
for input_seq_len, num_kv_pairs in [(128, 64), (256, 64), (512, 64), (1024, 64)]:
    if input_seq_len == 1024:
        batch_size = 64
    else:
        batch_size = 128

    factory_kwargs = {
        "num_kv_pairs": num_kv_pairs,
        "train_power_a": 0.01,
        "test_power_a": 0.01,
        "random_non_queries": False
    }

    data = DataConfig(
        train_configs=[MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        test_configs=[MQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        batch_size=batch_size,
        cache_dir=CACHE_DIR,
    )

    for d_model in [128]:
        for lr in  np.logspace(-4, -2, 4):
            
            # Mixer Definitions with dynamic head alignment
            MIXERS = {
                "GatedDeltaNet": dict(
                    name="zoology.mixers.gated_delta_net.GatedDeltaNet",   
                    kwargs={
                        "head_dim": d_model // 4, 
                        "num_heads": 4,           
                        "expand_v": 2,            
                        "mode": "chunk"
                    }
                ),
                "Palimpsa": dict(
                    name="zoology.mixers.palimpsa.Palimpsa",
                    kwargs={
                        "head_dim": d_model // 4, 
                        "num_heads": 4,           
                        "expand_v": 2,            
                        "mode": "chunk"
                    }
                ),
                "NotPalimpsa": dict(
                    name="zoology.mixers.palimpsa.Palimpsa",
                    kwargs={
                        "head_dim": d_model // 4, 
                        "num_heads": 4,           
                        "expand_v": 2,            
                        "mode": "chunk",
                        "metaplasticity": False
                    }
                )
            }

            # Block Mapping (Ensure GDN doesn't run in a Palimpsa block)
            BLOCKS = {
                "GatedDeltaNet": "GatedDeltaNetBlock",
                "Palimpsa": "PalimpsaBlock",
                "NotPalimpsa": "PalimpsaBlock"
            }

            for sequence_mixer in ["Palimpsa", "GatedDeltaNet", "NotPalimpsa"]:
                
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type=BLOCKS[sequence_mixer],
                    max_position_embeddings=0,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=MIXERS[sequence_mixer],
                    state_mixer=dict(name="torch.nn.Identity", kwargs={})
                )

                run_timestamp = datetime.now().strftime("%m%d")
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=64,
                    run_id=f"{sequence_mixer}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr:.2e}-{run_timestamp}",
                    logger=LoggerConfig(
                        project_name="Palimpsa_MQAR",
                        entity=os.environ.get("WANDB_ENTITY")
                    )
                )
                configs.append(config)