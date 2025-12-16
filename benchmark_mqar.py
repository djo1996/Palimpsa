import torch
import argparse
import wandb
import os
import math
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# --- IMPORTS ---
try:
    from model_mqar import LanguageModel
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from model_mqar import LanguageModel

from config_mqar import MQAR_CONFIGS
from data.data_mqar.config import DataConfig
from data.data_mqar.associative_recall import MQARConfig
from data.data_mqar.utils import prepare_data

# =============================================================================
# 1. FAST JIT DATASET (Fixes the generation slowness)
# =============================================================================
class MQARIterableDataset(IterableDataset):
    """
    Generates MQAR samples on-the-fly (Just-In-Time).
    Eliminates the long wait before training starts.
    """
    def __init__(self, vocab_size, seq_len, num_kv_pairs, total_samples, batch_size, power_a=0.01):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.power_a = power_a

    def generate_sample(self):
        # 1. Generate KV pairs
        # Randomly select keys and values
        keys = np.random.choice(self.vocab_size, self.num_kv_pairs, replace=False)
        values = np.random.choice(self.vocab_size, self.num_kv_pairs, replace=True)

        # 2. Interleave them: k1, v1, k2, v2...
        kv_len = 2 * self.num_kv_pairs
        if kv_len + 1 > self.seq_len:
            raise ValueError(f"SeqLen {self.seq_len} too short for {self.num_kv_pairs} KV pairs")

        kv_interleaved = np.empty(kv_len, dtype=int)
        kv_interleaved[0::2] = keys
        kv_interleaved[1::2] = values

        # 3. Build Sequence
        # [k1, v1, ..., kn, vn, filler..., query]
        input_seq = np.random.choice(self.vocab_size, self.seq_len, replace=True)
        input_seq[:kv_len] = kv_interleaved

        # 4. Select Query
        if self.power_a > 0:
            probs = (np.arange(self.num_kv_pairs) + 1) ** (-self.power_a)
            probs = probs / probs.sum()
            idx = np.random.choice(self.num_kv_pairs, p=probs)
        else:
            idx = np.random.randint(self.num_kv_pairs)

        query_key = keys[idx]
        target_val = values[idx]

        input_seq[-1] = query_key

        # 5. Create Labels (Only predict last token)
        labels = np.full(self.seq_len, -100, dtype=int)
        labels[-1] = target_val
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed_offset = 0
        if worker_info is not None:
            seed_offset = worker_info.id
        
        # Seed numpy per worker
        np.random.seed((os.getpid() + seed_offset) % (2**32))
        
        # Yield infinite stream up to total limit
        count = 0
        while count < self.total_samples:
            yield self.generate_sample()
            count += 1
            
    def __len__(self):
        return self.total_samples

# =============================================================================
# 2. METRICS & ANALYSIS
# =============================================================================
def compute_accuracy(logits, labels, ignore_index=-100):
    preds = torch.argmax(logits, dim=-1)
    mask = labels != ignore_index
    correct = (preds == labels) & mask
    total_valid = mask.sum().float()
    return (correct.sum().float() / total_valid).item() if total_valid > 0 else 0.0

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss, total_acc, steps = 0, 0, 0
    for input_ids, labels, _ in dataloader:
        input_ids, labels = input_ids.cuda(), labels.cuda()
        output = model(input_ids, labels=labels)
        total_loss += output.loss.item()
        total_acc += compute_accuracy(output.logits, labels)
        steps += 1
    model.train()
    return total_loss / steps, total_acc / steps

@torch.no_grad()
def analyze_palimpsa_reconstruction(model, dataloader, args):
    """
    Palimpsa White-Box Analysis.
    Calculates reconstruction error of KV pairs from the final state.
    """
    print(f"\n[Analysis] Computing KV Reconstruction Error (Seq Len: {args.seq_len})...")
    model.eval()
    
    layer_errors = {}
    num_batches = 20 
    
    iterator = iter(dataloader)
    for _ in tqdm(range(num_batches), desc="Analyzing Layers"):
        try:
            input_ids, _, _ = next(iterator)
        except StopIteration:
            break
        input_ids = input_ids.cuda()

        # 1. Embeddings
        hidden_states = model.backbone.embeddings(input_ids)

        # 2. Layer Loop
        for i, layer in enumerate(model.backbone.layers):
            if i not in layer_errors: layer_errors[i] = []
            
            mixer = layer.mixer
            normed_states = layer.attn_norm(hidden_states)
            
            # --- Palimpsa Projections ---
            q = mixer.q_proj(normed_states)
            k = mixer.k_proj(normed_states)
            v = mixer.v_proj(normed_states)
            
            # Conv
            if mixer.use_short_conv:
                q, _ = mixer.q_conv1d(q, None)
                k, _ = mixer.k_conv1d(k, None)
                v, _ = mixer.v_conv1d(v, None)
            
            # Dimensions
            head_k_dim = mixer.head_k_dim
            head_v_dim = mixer.head_v_dim
            
            q = rearrange(q, '... (h d) -> ... h d', d=head_k_dim)
            k = rearrange(k, '... (h d) -> ... h d', d=head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=head_v_dim)
            
            # Beta & Gating
            b = mixer.b_proj(mixer.b_rank_proj(normed_states)).float()
            b = rearrange(b, '... (h d) -> ... h d', d=head_v_dim)
            
            bs = torch.sigmoid(mixer.bs_proj(normed_states).float())
            b = torch.sigmoid(b).to(normed_states.dtype) * bs.unsqueeze(-1)
            
            # Apply Beta to Value
            v_target = v * b

            # Softmax Q, K
            q = F.softmax(q, dim=-1)
            k = F.softmax(k, dim=-1)
            
            # --- Get Final State ---
            dt = F.softplus(mixer.dt_proj(normed_states).float() + mixer.dt_bias)
            Ip = torch.exp(mixer.Ip_log.float())
            A = mixer.A_log.float().exp()
            
            from palimpsa.ops.palimpsa import chunk_palimpsa
            _, final_mu, _ = chunk_palimpsa(
                q=q, k=k, v=v_target, b=b, gt=dt, g=A, Ip=Ip,
                scale=1.0, output_final_state=True
            )
            
            # --- FIXED EINSUM ---
            # Palimpsa State (final_mu) shape is (Batch, Head, V, K) [Transposed state]
            # K shape is (Batch, Seq, Head, K)
            # Reconstruction = State @ K.T -> (V, K) @ (K, 1) -> V
            
            # b: batch, h: head, v: val_dim, k: key_dim, l: seq_len
            reconstruction = einsum(final_mu, k, "b h v k, b l h k -> b l h v")
            
            diff_sq = (reconstruction - v_target) ** 2
            weighted_error = b * diff_sq # Weighted by beta
            token_error = weighted_error.sum(dim=(-2, -1)).mean()
            
            layer_errors[i].append(token_error.item())

            # Continue forward
            hidden_states, _ = layer(hidden_states, None)

    print("\n" + "="*40)
    print(f"   KV Reconstruction Error (Beta-Weighted)")
    print("="*40)
    for i in sorted(layer_errors.keys()):
        avg_err = sum(layer_errors[i]) / len(layer_errors[i])
        print(f"Layer {i:02d}: {avg_err:.6f}")
        if args.use_wandb:
            wandb.log({f"analysis/layer_{i}_kv_recon_error": avg_err})
    print("="*40 + "\n")
    model.train()

# =============================================================================
# 3. MAIN TRAINING LOOP
# =============================================================================
def run_training(args, current_seq_len):
    # Auto-Batch Size
    if args.batch_size is None:
        if current_seq_len >= 1024: batch_size = 64
        elif current_seq_len >= 512: batch_size = 128
        elif current_seq_len >= 256: batch_size = 256
        else: batch_size = 512
    else:
        batch_size = args.batch_size

    # Auto-Steps (Match Zoology 6.4M tokens)
    target_examples = 6_400_000
    total_steps = int(target_examples / batch_size)
    steps_to_run = args.steps if args.steps > 0 else total_steps

    print(f"\n=== Run: {args.config} | Len {current_seq_len} | Batch {batch_size} | Steps {steps_to_run} ===")

    # 1. Config
    if args.config not in MQAR_CONFIGS: raise ValueError(f"Config '{args.config}' not found.")
    model_config = MQAR_CONFIGS[args.config]
    model_config.max_position_embeddings = current_seq_len
    model_config.vocab_size = args.vocab_size
    model_config.d_model = args.d_model
    
    # 2. Fast Training Data
    print("Initializing Fast Data Generator...")
    train_dataset = MQARIterableDataset(
        vocab_size=args.vocab_size,
        seq_len=current_seq_len,
        num_kv_pairs=args.num_kv_pairs,
        total_samples=steps_to_run * batch_size,
        batch_size=batch_size
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # 3. Test Data (Standard Static)
    print("Preparing Test Set...")
    mqar_conf_test = MQARConfig(
        vocab_size=args.vocab_size, input_seq_len=current_seq_len,
        num_examples=3000, num_kv_pairs=args.num_kv_pairs, power_a=0.01 
    )
    test_data_config = DataConfig(train_configs=[], test_configs=[mqar_conf_test], batch_size=batch_size, cache_dir=args.cache_dir)
    _, test_loader = prepare_data(test_data_config)

    # 4. WandB
    run_name = f"{args.config}_L{current_seq_len}_D{args.d_model}_KV{args.num_kv_pairs}"
    if args.use_wandb:
        wandb.init(
            project="Palimpsa_MQAR_Benchmark", group=args.config, name=run_name,
            config={"seq_len": current_seq_len, "batch_size": batch_size, "steps": steps_to_run, **args.__dict__},
            reinit=True
        )

    # 5. Model
    model = LanguageModel(model_config).cuda()
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_to_run, eta_min=0.0)

    # 6. Train
    model.train()
    pbar = tqdm(train_loader, total=steps_to_run, desc=run_name)
    best_val_acc = 0.0

    # Note: IterableDataset yields batches directly, no 'slices'
    for step, (input_ids, labels) in enumerate(pbar):
        if step >= steps_to_run: break
        
        input_ids, labels = input_ids.cuda(), labels.cuda()
        output = model(input_ids, labels=labels)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 50 == 0:
            train_acc = compute_accuracy(output.logits, labels)
            pbar.set_description(f"{run_name} | Loss: {loss.item():.4f} | Acc: {train_acc:.1%}")
            if args.use_wandb:
                wandb.log({"train/loss": loss.item(), "train/acc": train_acc, "train/lr": scheduler.get_last_lr()[0], "step": step})

        if step % 500 == 0 and step > 0:
            val_loss, val_acc = evaluate(model, test_loader)
            if val_acc > best_val_acc: best_val_acc = val_acc
            if args.use_wandb: wandb.log({"val/loss": val_loss, "val/acc": val_acc, "step": step})

    print(f"Run Finished. Best Val Acc: {best_val_acc:.2%}")

    # 7. Analysis
    if args.compute_inner_loss and args.config == 'palimpsa':
        try:
            analyze_palimpsa_reconstruction(model, test_loader, args)
        except Exception as e:
            print(f"Analysis failed with error: {e}")
    
    if args.use_wandb: wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="palimpsa", choices=["palimpsa", "gla", "gated_deltanet"])
    parser.add_argument("--seq_len", type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_kv_pairs", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="./data_cache")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--compute_inner_loss", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Fix manual arg update for seq_len in analysis
    seq_lens = args.seq_len
    for s in seq_lens:
        args.seq_len = s # Update single value for the run
        run_training(args, s)