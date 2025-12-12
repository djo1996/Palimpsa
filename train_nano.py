"""
train_nano.py
A minimal, single-GPU training script for Palimpsa.
Adapted from Karpathy's nanoGPT & Longhorn.

Usage:
$ python train_nano.py
"""

import os
import time
import math
import pickle
import numpy as np
import torch
from contextlib import nullcontext

# --- Import Palimpsa ---
# Assuming you ran `pip install -e .`, these imports will work.
# If you are hacking on the files directly, ensure they are in the python path.
from palimpsa.models.palimpsa.configuration_palimpsa import PalimpsaConfig
from palimpsa.models.palimpsa.modeling_palimpsa import PalimpsaForCausalLM

# -----------------------------------------------------------------------------
# 1. Configuration (Edit these directly)
# -----------------------------------------------------------------------------
out_dir = 'out_nano'
eval_interval = 250
log_interval = 10
eval_iters = 200
always_save_checkpoint = False

# Data
dataset = 'shakespeare'
batch_size = 64
block_size = 256  # Context length

# Model (Mini Palimpsa for Shakespeare)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# Optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # use PyTorch 2.0 to compile the model

# -----------------------------------------------------------------------------
# 2. Setup & Data Loading
# -----------------------------------------------------------------------------
torch.manual_seed(1337)
os.makedirs(out_dir, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)

print(f"Training on {device} using {dtype}")

# Load Data
data_dir = os.path.join('data', dataset)
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data not found at {data_dir}. Did you run 'python prepare.py' in data/shakespeare?")

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# 3. Model Initialization
# -----------------------------------------------------------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = 50304
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size}")

print("Initializing Palimpsa...")
config = PalimpsaConfig(
    vocab_size=meta_vocab_size,
    hidden_size=n_embd,
    num_hidden_layers=n_layer,
    num_heads=n_head,
    num_kv_heads=n_head, 
    max_position_embeddings=block_size,
    fuse_cross_entropy=True, # Uses the fused kernel from your modeling code
    use_cache=False,
    # Palimpsa specific defaults
    expand_v=1.0,
    expand_k=1.0, 
)

model = PalimpsaForCausalLM(config)
model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# -----------------------------------------------------------------------------
# 4. Optimizer
# -----------------------------------------------------------------------------
# Separate weight decay params (matmuls) from no-decay params (biases, layernorms, dt_bias)
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))

if compile:
    print("Compiling model... (this might take a minute)")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                outputs = model(X, Y)
                loss = outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Scheduler
def get_lr(it):
    warmup_iters = 200
    lr_decay_iters = max_iters
    min_lr = learning_rate / 10
    
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

iter_num = 0
best_val_loss = 1e9
t0 = time.time()

while True:
    # 1. Update LR
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2. Evaluate
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']

    # 3. Forward + Backward
    X, Y = get_batch('train')
    with ctx:
        outputs = model(X, labels=Y)
        loss = outputs.loss

    # Scaler/Backprop logic handled automatically by pytorch usually, 
    # but strictly speaking for fp16 we might want scaler. 
    # Keeping it simple for BF16 (Ampere+) which doesn't need scaler.
    loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # 4. Log
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")

    iter_num += 1
    if iter_num > max_iters:
        break