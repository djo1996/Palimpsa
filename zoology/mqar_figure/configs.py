import uuid
import os
import numpy as np
import torch
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig
from datetime import datetime

# =================================================================
# 1. Environment & Seed Setup
# =================================================================
SWEEP_SEED = 4
def set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

configs = []
for SWEEP_SEED in [1,2,3,4,5]:

    set_seed(SWEEP_SEED)

    sweep_id = uuid.uuid4().hex[:6]
    VOCAB_SIZE = 8_192

    # Use a local cache directory relative to the project root
    # This avoids "Could not create directory" permission errors on the cluster
    CACHE_DIR = os.path.join(os.getcwd(), f"cache/mqar_seed_{SWEEP_SEED}")
    for input_seq_len, num_kv_pairs in [(128, 32), (256, 64), (512, 128), (1024, 256)]:
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
            seed=SWEEP_SEED,
        )
        for d_model in [128]:
            for lr in [2.15e-3, 1e-2]:
                # in  np.logspace(-4, -2, 4)[2.15e-3, 1e-2]
                # Mixer Definitions with dynamic head alignment
                MIXERS = {
                    "GatedDeltaNet": dict(
                        name="zoology.mixers.gated_delta_net.GatedDeltaNet",   
                        kwargs={
                            "head_dim": d_model // 8, 
                            "num_heads": 8,           
                            "expand_v": 2,            
                            "mode": "chunk"
                        }
                    ),
                    "Palimpsa": dict(
                        name="zoology.mixers.palimpsa.Palimpsa",
                        kwargs={
                            "head_dim": d_model // 8, 
                            "num_heads": 8,           
                            "expand_v": 2,   
                            "mode": "chunk",
                            "beta_step_rank": d_model // 16 ,
                            "qk_act": "siluL2"
                        }
                    ),
                    "NotPalimpsa": dict(
                        name="zoology.mixers.palimpsa.Palimpsa",
                        kwargs={
                            "head_dim": d_model // 8, 
                            "num_heads": 8,           
                            "expand_v": 2, 
                            "mode": "chunk",
                            "metaplasticity": False,
                            "beta_step_rank": d_model // 16 ,
                            "qk_act": "siluL2"
                        }
                    ),
                    "MetaMamba2": dict(
                        name="zoology.mixers.meta_mamba2.MetaMamba2",
                        kwargs={
                            "state_size": d_model // 8, 
                            "head_dim": d_model // 4, 
                            "num_heads": 8,  
                            "n_groups": 8,        
                            "expand": 2,            
                            "beta_step_rank": d_model // 16 ,
                            "mode": "chunk"
                        }
                    ),
                    "Mamba2": dict(
                        name="zoology.mixers.meta_mamba2.MetaMamba2",
                        kwargs={
                            "state_size": d_model // 8, 
                            "head_dim": d_model // 4,
                            "num_heads": 8,  
                            "n_groups": 8,      
                            "expand": 2,            
                            "mode": "chunk",
                            "beta_step_rank": d_model // 16 ,
                            "metaplasticity": False,
                        }
                    )
                }

                # Block Mapping (Ensure GDN doesn't run in a Palimpsa block)
                BLOCKS = {
                    "GatedDeltaNet": "GatedDeltaNetBlock",
                    "Palimpsa": "PalimpsaBlock",
                    "NotPalimpsa": "PalimpsaBlock",
                    "MetaMamba2": "MetaMamba2Block",
                    "Mamba2": "MetaMamba2Block"
                }

                for sequence_mixer in ["Palimpsa","MetaMamba2", "NotPalimpsa", "Mamba2", "GatedDeltaNet"]:
                    
                    model = ModelConfig(
                        d_model=d_model,
                        n_layers=2,
                        block_type=BLOCKS[sequence_mixer],
                        max_position_embeddings=0,
                        vocab_size=VOCAB_SIZE,
                        sequence_mixer=MIXERS[sequence_mixer],
                        state_mixer=dict(name="torch.nn.Identity", kwargs={}),
                        forgetting = 'small'
                    )

                    run_timestamp = datetime.now().strftime("%m%d")
                    config = TrainConfig(
                        model=model,
                        data=data,
                        learning_rate=lr,
                        max_epochs=64,
                        seed=SWEEP_SEED,
                        run_id=f"{sequence_mixer}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr:.2e}-seed{SWEEP_SEED}-{run_timestamp}",
                        logger=LoggerConfig(
                            project_name="Palimpsa_MQAR_seeds_very_small_forgetting-2",
                            entity=os.environ.get("WANDB_ENTITY")
                        )
                    )
                    configs.append(config)