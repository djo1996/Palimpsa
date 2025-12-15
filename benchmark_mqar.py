# Save this as bench_mqar.py in the root of your Palimpsa repo
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

# Adjust these imports to match your new clean package structure
# e.g., if you have Palimpsa/models/palimpsa.py
try:
    from palimpsa import PalimpsaConfig, PalimpsaForCausalLM
except ImportError:
    # Fallback if package isn't installed in editable mode yet
    import sys
    sys.path.append(".") 
    from palimpsa.models import PalimpsaConfig, PalimpsaForCausalLM

class MQARGenerator:
    """
    On-the-fly generator for Multi-Query Associative Recall.
    No complex dataloaders, just raw tensors.
    """
    def __init__(self, vocab_size, seq_len, num_kv_pairs, batch_size, device='cuda'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        while True:
            # 1. Init sequences with random noise
            input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
            labels = input_ids.clone()
            
            # 2. Generate Keys and Values
            # We reserve the first 'num_kv_pairs' * 2 positions for the KV definitions? 
            # Or scatter them? For standard MQAR, we usually scatter them.
            # Here is a simplified version: Puts KV pairs at the start, queries at the end.
            
            keys = torch.randint(0, self.vocab_size, (self.batch_size, self.num_kv_pairs), device=self.device)
            values = torch.randint(0, self.vocab_size, (self.batch_size, self.num_kv_pairs), device=self.device)
            
            # Place K V K V ... at the beginning
            for i in range(self.num_kv_pairs):
                input_ids[:, 2*i] = keys[:, i]     # Key
                input_ids[:, 2*i+1] = values[:, i] # Value
                # We don't want to predict the Key, we want to predict the Value
                labels[:, 2*i] = -100 
            
            # Place Queries at the end (repeat the keys)
            # This is a basic "recall" setup.
            # In a real rigorous test, you might shuffle positions.
            start_query_idx = self.seq_len - self.num_kv_pairs
            input_ids[:, start_query_idx:] = keys
            labels[:, start_query_idx:] = values
            
            # Mask everything else in labels
            labels[:, 2*self.num_kv_pairs : start_query_idx] = -100

            yield input_ids, labels

def train(args):
    print(f"--- MQAR Benchmark: {args.model} | SeqLen {args.seq_len} | KV {args.num_kv_pairs} ---")
    
    config = PalimpsaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.d_model,
        num_hidden_layers=2,
        max_position_embeddings=args.seq_len
    )
    
    if args.model == 'palimpsa':
        model = PalimpsaForCausalLM(config).cuda()
    elif args.model == 'gla':
        from fla.models import GatedLinearAttention
        model = GatedLinearAttention(config).cuda()

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
        
        pbar.set_description(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="palimpsa", choices=["palimpsa", "gla"])
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_kv_pairs", type=int, default=16)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    
    train(args)