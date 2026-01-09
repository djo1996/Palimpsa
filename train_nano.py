"""
train_nano.py
Minimal training script for Palimpsa & FLA baselines on Shakespeare.
Usage: python train_nano.py --model palimpsa --batch_size 64
"""
import os
import time
import pickle
import argparse
import numpy as np
import torch
from contextlib import nullcontext

# --- Imports ---
from palimpsa.models.palimpsa.configuration_palimpsa import PalimpsaConfig
from palimpsa.models.palimpsa.modeling_palimpsa import PalimpsaForCausalLM
from palimpsa.models.meta_mamba2.configuration_meta_mamba2 import MetaMamba2Config
from palimpsa.models.meta_mamba2.modeling_met_mamba2 import MetaMamba2ForCausalLM
# Import FLA baselines dynamically to avoid crashing if FLA isn't installed
try:
    from fla.models import GLAForCausalLM, GLAConfig
    from fla.models import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
except ImportError:
    print("Warning: flash-linear-attention not found. Baselines (gla, gated_deltanet) will fail.")

def get_args():
    parser = argparse.ArgumentParser(description="Train Palimpsa/FLA on Shakespeare")
    parser.add_argument("--model", type=str, default="palimpsa", choices=["palimpsa", "meta_mamba2", "gla", "gated_deltanet"], help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--block_size", type=int, default=256, help="Context length")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of heads")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_iters", type=int, default=5000, help="Max iterations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

args = get_args()

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
torch.manual_seed(1337)
out_dir = 'out_nano'
os.makedirs(out_dir, exist_ok=True)
data_dir = os.path.join('data', 'shakespeare_char')

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory {data_dir} not found. Did you run prepare.py?")

# Config
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if args.device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

print(f"Training {args.model} on {args.device} | Batch: {args.batch_size} | Context: {args.block_size}")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+args.block_size]).astype(np.int64)) for i in ix])
    if args.device == 'cuda':
        x, y = x.pin_memory().to(args.device, non_blocking=True), y.pin_memory().to(args.device, non_blocking=True)
    else:
        x, y = x.to(args.device), y.to(args.device)
    return x, y

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = 65
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

MODEL_REGISTRY = {
    "palimpsa": (PalimpsaConfig, PalimpsaForCausalLM),
    "gla": (GLAConfig, GLAForCausalLM),
    "gated_deltanet": (GatedDeltaNetConfig, GatedDeltaNetForCausalLM),
}

print(f"Initializing {args.model}...")
ConfigClass, ModelClass = MODEL_REGISTRY[args.model]

# Common config args
config_args = dict(
    vocab_size=meta_vocab_size,
    hidden_size=args.n_embd,
    num_hidden_layers=args.n_layer,
    num_heads=args.n_head,
    head_dim= args.n_embd // args.n_head,
    max_position_embeddings=args.block_size,
    use_cache=False,
    expand_v = 2,
    expand_k = 1
)



config = ConfigClass(**config_args)
model = ModelClass(config)
model.to(args.device)

print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-1)

# -----------------------------------------------------------------------------
# Train Loop
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            with ctx:
                outputs = model(X, labels=Y)
            losses[k] = outputs.loss.item()
        out[split] = losses.mean()
    model.train()
    return out

iter_num = 0
t0 = time.time()

while True:
    if iter_num % 250 == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    X, Y = get_batch('train')
    with ctx:
        outputs = model(X, labels=Y)
        loss = outputs.loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if iter_num % 50 == 0:
        t1 = time.time()
        dt = (t1 - t0) * 1000
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt:.2f}ms")

    iter_num += 1
    if iter_num > args.max_iters:
        break