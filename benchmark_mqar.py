import torch
import torch.nn as nn
import argparse
import numpy as np
import wandb
from tqdm import tqdm
from contextlib import nullcontext

# --- 1. Robust Imports (Matching your file structure) ---
try:
    # Try clean import first (in case you fix __init__.py later)
    from palimpsa.models.palimpsa.configuration_palimpsa import PalimpsaConfig
    from palimpsa.models.palimpsa.modeling_palimpsa import PalimpsaForCausalLM
except ImportError:
    # Fallback to local path if running from root without package install
    import sys, os
    sys.path.append(os.getcwd())
    from palimpsa.models.palimpsa.configuration_palimpsa import PalimpsaConfig
    from palimpsa.models.palimpsa.modeling_palimpsa import PalimpsaForCausalLM

# --- 2. FLA Baselines (Dynamic Import) ---
try:
    from fla.models import GLAForCausalLM, GLAConfig
    from fla.models import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("⚠️ FLA not installed. GLA and GatedDeltaNet will be unavailable.")

# --- 3. The Real MQAR Logic (No external dependency) ---
class MQARGenerator:
    """
    Multi-Query Associative Recall Generator.
    - Scatters KV pairs randomly in the context.
    - Queries are appended at the end.
    - Crucial for testing 'needle-in-haystack' capability.
    """
    def __init__(self, vocab_size, seq_len, num_kv_pairs, batch_size, device='cuda'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        while True:
            # Init empty inputs and labels
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
            labels = torch.full_like(input_ids, -100)
            
            # Generate Keys and Values
            keys = torch.randint(0, self.vocab_size, (self.batch_size, self.num_kv_pairs), device=self.device)
            values = torch.randint(0, self.vocab_size, (self.batch_size, self.num_kv_pairs), device=self.device)
            
            # --- Scatter KV pairs ---
            # We need 2 positions per pair (Key, Value) + space for queries at end
            # We reserve the last `num_kv_pairs` tokens for queries
            context_len = self.seq_len - self.num_kv_pairs
            
            for b in range(self.batch_size):
                # Select random indices for Keys (ensure space for Value after each Key)
                # We simply pick 'num_kv_pairs' indices from 0 to context_len-2
                # This is a simplified scatter: strictly K then V immediately
                possible_indices = np.arange(context_len - 1)
                kv_start_indices = np.random.choice(possible_indices, self.num_kv_pairs, replace=False)
                
                # Assign K and V
                input_ids[b, kv_start_indices] = keys[b]
                input_ids[b, kv_start_indices + 1] = values[b]
                
                # --- Create Queries at the end ---
                # The last chunk is just the keys again
                input_ids[b, context_len:] = keys[b]
                
                # The target for the queries is the values
                labels[b, context_len:] = values[b]

            yield input_ids, labels

# --- 4. Training Loop ---
def train(args):
    print(f"--- MQAR Benchmark: {args.model} | SeqLen {args.seq_len} | KV {args.num_kv_pairs} ---")
    
    if args.use_wandb:
        wandb.init(project="Palimpsa_MQAR", config=args, name=f"{args.model}_kv{args.num_kv_pairs}")

    # Configuration map
    MODEL_MAP = {
        "palimpsa": (PalimpsaConfig, PalimpsaForCausalLM),
    }
    if FLA_AVAILABLE:
        MODEL_MAP.update({
            "gla": (GLAConfig, GLAForCausalLM),
            "gated_deltanet": (GatedDeltaNetConfig, GatedDeltaNetForCausalLM)
        })

    if args.model not in MODEL_MAP:
        raise ValueError(f"Model {args.model} not implemented or FLA missing.")

    ConfigClass, ModelClass = MODEL_MAP[args.model]
    
    # Init Config
    config = ConfigClass(
        vocab_size=args.vocab_size,
        hidden_size=args.d_model,
        num_hidden_layers=2, # Small model for MQAR usually enough
        num_heads=4, 
        max_position_embeddings=args.seq_len,
        use_cache=False
    )
    
    model = ModelClass(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = MQARGenerator(args.vocab_size, args.seq_len, args.num_kv_pairs, args.batch_size)
    data_iter = iter(generator)
    
    pbar = tqdm(range(args.steps))
    for step in pbar:
        input_ids, labels = next(data_iter)
        
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    args = parser.parse_args()
    
    train(args)