
import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, DataSegmentConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig
import torch
# =================================================================
# 1. Define a global seed for the entire sweep of experiments
# =================================================================
SWEEP_SEED = 3

# You can also use this utility function for extra certainty
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call it at the beginning of your script
set_seed(SWEEP_SEED)

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "figure2" + sweep_id


VOCAB_SIZE = 8_192


configs = []
for input_seq_len, num_kv_pairs in [
    # (128, 8),
    # (64, 4),
    # (256, 16),
    (512, 64),
    (1024, 128),
]:
    if input_seq_len == 1024:
        batch_size = 64
    elif input_seq_len == 512:
        batch_size = 128
    elif input_seq_len == 256:
        batch_size = 256
    else:
        batch_size = 512


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
        cache_dir="/var/cr05_data/sabri_data/zoology",
        # cache_dir="", # TODO: add a directory to cache your data!
    )

    for d_model in [
         128
    ]:
        for lr in  [0.01]:
            
            MIXERS = {
                "GatedDeltaNet": dict(
                    name="zoology.mixers.gated_delta_net.GatedDeltaNet",   
                    wargs={
                        "head_dim": d_model // 4, 
                        "num_heads": 4,           
                        "expand_v": 1,            
                        "mode": "chunk"
                    }
                ),
                "Palimpsa": dict(
                    name="zoology.mixers.palimpsa.Palimpsa",
                    kwargs={
                        "head_dim": d_model // 4, 
                        "num_heads": 4,           
                        "expand_v": 1,            
                        "mode": "chunk"
                    }
                )
            }

            for sequence_mixer in [
                "Palimpsa", "GatedDeltaNet"
            ]:
                block_type = "PalimpsaBlock"
                print(block_type)
                
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    block_type=block_type,
                    max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=MIXERS[sequence_mixer],
                    state_mixer=dict(name="torch.nn.Identity", kwargs={})
                )
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=64,
                    run_id=f"{sequence_mixer}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr}-kv{num_kv_pairs}",
                    logger=LoggerConfig(
                        project_name="zoology",
                        entity="hazy-research"
                    )

                )
                configs.append(config)