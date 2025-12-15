import torch
import argparse
import wandb
import os
from tqdm import tqdm

# --- IMPORTS ---
try:
    from palimpsa import PalimpsaConfig, PalimpsaForCausalLM
except ImportError:
    # Fallback to local
    import sys
    sys.path.append(os.getcwd())
    from palimpsa.models.palimpsa.configuration_palimpsa import PalimpsaConfig
    from palimpsa.models.palimpsa.modeling_palimpsa import PalimpsaForCausalLM

# Import the New Data Pipeline
from palimpsa.data.data_mqar.config import DataConfig
from palimpsa.data.data_mqar.associative_recall import MQARConfig
from palimpsa.data.data_mqar.utils import prepare_data

# FLA Baselines
try:
    from fla.models import GLAForCausalLM, GLAConfig
    from fla.models import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("⚠️ FLA not found. Baselines unavailable.")

def train(args):
    print(f"--- MQAR Benchmark: {args.model} | SeqLen {args.seq_len} | KV {args.num_kv_pairs} ---")
    
    # 1. Setup Data Config (Zoology Style)
    mqar_conf = MQARConfig(
        vocab_size=args.vocab_size,
        input_seq_len=args.seq_len,
        num_examples=args.steps * args.batch_size, # Generate enough data for the steps
        num_kv_pairs=args.num_kv_pairs,
        power_a=0.01 # Zipfian distribution for gaps
    )
    
    data_config = DataConfig(
        train_configs=[mqar_conf],
        test_configs=[mqar_conf], # Just reuse for simplified bench script
        batch_size=args.batch_size,
        cache_dir=os.path.expanduser(args.cache_dir) if args.cache_dir else None
    )

    print("Preparing Data (this might take a moment to generate/cache)...")
    train_loader, _ = prepare_data(data_config)

    # 2. Setup Model
    if args.use_wandb:
        wandb.init(project="Palimpsa_MQAR", config=args, name=f"{args.model}_kv{args.num_kv_pairs}")

    if args.model == "palimpsa":
        config = PalimpsaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.d_model,
            num_hidden_layers=2,
            max_position_embeddings=args.seq_len,
            use_cache=False
        )
        model = PalimpsaForCausalLM(config).cuda()
    elif FLA_AVAILABLE and args.model == "gla":
        config = GLAConfig(vocab_size=args.vocab_size, hidden_size=args.d_model, max_position_embeddings=args.seq_len)
        model = GLAForCausalLM(config).cuda()
    elif FLA_AVAILABLE and args.model == "gated_deltanet":
        config = GatedDeltaNetConfig(vocab_size=args.vocab_size, hidden_size=args.d_model, max_position_embeddings=args.seq_len)
        model = GatedDeltaNetForCausalLM(config).cuda()
    else:
        raise ValueError(f"Model {args.model} not found or FLA missing.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 3. Training Loop
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader))
    
    for step, (input_ids, labels, slices) in enumerate(pbar):
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if step % 10 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}")
            if args.use_wandb:
                wandb.log({"loss": loss.item(), "step": step})
        
        if step >= args.steps:
            break

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="palimpsa", choices=["palimpsa", "gla", "gated_deltanet"])
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_kv_pairs", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--cache_dir", type=str, default="./data_cache", help="Where to save synthetic data")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    train(args)