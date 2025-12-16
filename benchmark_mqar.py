import torch
import argparse
import wandb
import os
import math
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat, einsum
import torch.nn.functional as F

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

# --- METRICS ---
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

# --- PALIMPSA WHITE-BOX ANALYSIS ---
@torch.no_grad()
def analyze_palimpsa_reconstruction(model, dataloader, args):
    """
    Manual reconstruction analysis for Palimpsa Layers.
    Replicates the forward pass logic to extract K, V, Beta and Final State (Mu).
    """
    print(f"\n[Analysis] Computing KV Reconstruction Error (Seq Len: {args.seq_len})...")
    model.eval()
    
    # Storage for error metrics per layer
    layer_errors = {}
    
    # We only need a few batches to get a statistically significant number
    num_batches = 20 
    
    iterator = iter(dataloader)
    for _ in tqdm(range(num_batches), desc="Analyzing Layers"):
        try:
            input_ids, _, _ = next(iterator)
        except StopIteration:
            break
            
        input_ids = input_ids.cuda()
        
        # 1. Get Initial Embeddings
        hidden_states = model.backbone.embeddings(input_ids)

        # 2. Iterate Layers
        for i, layer in enumerate(model.backbone.layers):
            if i not in layer_errors: layer_errors[i] = []
            
            # Helper to access the inner Palimpsa module
            # Logic: UnifiedBlock -> self.mixer (Palimpsa)
            mixer = layer.mixer
            
            # --- REPLICATE PALIMPSA FORWARD PASS ---
            # A. Pre-Norm (UnifiedBlock handles this)
            normed_states = layer.attn_norm(hidden_states)
            
            # B. Projections (q, k, v)
            q = mixer.q_proj(normed_states)
            k = mixer.k_proj(normed_states)
            v = mixer.v_proj(normed_states)
            
            # C. Short Convolution
            # We must use the layer's internal conv modules
            if mixer.use_short_conv:
                # Note: We pass None as cache to process whole sequence at once
                q, _ = mixer.q_conv1d(q, None)
                k, _ = mixer.k_conv1d(k, None)
                v, _ = mixer.v_conv1d(v, None)
            
            # D. Beta (Decay/Gate) Projection
            # Matches: b = self.b_proj(self.b_rank_proj(hidden_states))
            b = mixer.b_proj(mixer.b_rank_proj(normed_states)).float()
            
            # E. Reshaping (Head splitting)
            # Shapes: [Batch, Seq, n_heads * head_dim] -> [Batch, Seq, n_heads, head_dim]
            head_k_dim = mixer.head_k_dim
            head_v_dim = mixer.head_v_dim
            
            q = rearrange(q, '... (h d) -> ... h d', d=head_k_dim)
            k = rearrange(k, '... (h d) -> ... h d', d=head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=head_v_dim)
            b = rearrange(b, '... (h d) -> ... h d', d=head_v_dim)

            # Handle Grouped Query Attention (GQA) broadcasting if needed
            if mixer.num_v_heads > mixer.num_heads:
                g = mixer.num_v_heads // mixer.num_heads
                q = repeat(q, '... h d -> ... (h g) d', g=g)
                k = repeat(k, '... h d -> ... (h g) d', g=g)

            # F. Beta Activations
            bs = torch.sigmoid(mixer.bs_proj(normed_states).float())
            
            # Exact logic from Palimpsa code:
            b = torch.sigmoid(b).to(normed_states.dtype)
            b = b * bs.unsqueeze(-1)
            
            # G. Value Gating (The target V is modified by Beta in Palimpsa)
            # We keep 'v_target' as the value that actually enters the kernel
            v_target = v * b 

            # H. Softmax Normalization (Critical for Palimpsa)
            q = F.softmax(q, dim=-1)
            k = F.softmax(k, dim=-1)
            
            # --- GET FINAL STATE (S_T) ---
            # We call the kernel wrapper to get the final state.
            # We need the other params (dt, A, Ip) required by chunk_palimpsa
            
            dt = F.softplus(mixer.dt_proj(normed_states).float() + mixer.dt_bias)
            Ip = torch.exp(mixer.Ip_log.float())
            A = mixer.A_log.float().exp()

            from palimpsa.ops.palimpsa import chunk_palimpsa
            
            # Call kernel to get final state
            # Returns: o, final_mu, final_I
            _, final_mu, _ = chunk_palimpsa(
                q=q, k=k, v=v_target, b=b, gt=dt, g=A, Ip=Ip,
                scale=1.0, output_final_state=True
            )
            
            # --- CALCULATE ERROR ---
            # final_mu shape: (Batch, Head, Dim_K, Dim_V) -> (B, H, D, D) usually
            # k shape:        (Batch, Seq, Head, Dim_K)
            # v_target shape: (Batch, Seq, Head, Dim_V)
            
            # Reconstruction: S_T @ k_t^T
            # Einsum: 
            #   S: b h k v (State)
            #   K: b l h k (Key at timestep l)
            #   -> b l h v (Reconstructed Value)
            
            reconstruction = einsum(final_mu, k, "b h k v, b l h k -> b l h v")
            
            # Error Formula: sum( beta * (Recon - Target)**2 )
            # note: 'b' is our beta tensor
            
            diff_sq = (reconstruction - v) ** 2
            weighted_error = b * diff_sq
            
            # Average error per token
            # Sum over (Head, Dim), Mean over (Batch, Seq)
            token_error = weighted_error.sum(dim=(-2, -1)).mean()
            
            layer_errors[i].append(token_error.item())

            # --- CONTINUE FORWARD PASS ---
            # We must run the real forward pass to prep inputs for next layer
            hidden_states, _ = layer(hidden_states, None) # Residual passed as None for simplicity here

    # Log Results
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


# --- MAIN BENCHMARK SCRIPT ---
def run_training(args, current_seq_len):
    # 1. Dynamic Batch Sizing
    if args.batch_size is None:
        if current_seq_len >= 1024: batch_size = 64
        elif current_seq_len >= 512: batch_size = 128
        elif current_seq_len >= 256: batch_size = 256
        else: batch_size = 512
    else:
        batch_size = args.batch_size

    # Zoology-matching steps calculation
    target_examples = 6_400_000
    total_steps = int(target_examples / batch_size)
    steps_to_run = args.steps if args.steps > 0 else total_steps

    print(f"\n=== Run: {args.config} | Len {current_seq_len} | Batch {batch_size} | Steps {steps_to_run} ===")

    # 2. Config & Data
    if args.config not in MQAR_CONFIGS: raise ValueError(f"Config '{args.config}' not found.")
    
    model_config = MQAR_CONFIGS[args.config]
    model_config.max_position_embeddings = current_seq_len
    model_config.vocab_size = args.vocab_size
    model_config.d_model = args.d_model
    
    mqar_conf = MQARConfig(
        vocab_size=args.vocab_size, input_seq_len=current_seq_len,
        num_examples=steps_to_run * batch_size, num_kv_pairs=args.num_kv_pairs, power_a=0.01 
    )
    test_conf = mqar_conf.model_copy()
    test_conf.num_examples = 3000

    data_config = DataConfig(train_configs=[mqar_conf], test_configs=[test_conf], batch_size=batch_size, cache_dir=args.cache_dir)
    train_loader, test_loader = prepare_data(data_config)

    # 3. WandB
    run_name = f"{args.config}_L{current_seq_len}_D{args.d_model}_KV{args.num_kv_pairs}"
    if args.use_wandb:
        wandb.init(
            project="Palimpsa_MQAR_Benchmark", group=args.config, name=run_name,
            config={"seq_len": current_seq_len, "batch_size": batch_size, "steps": steps_to_run, **args.__dict__},
            reinit=True
        )

    # 4. Model & Optim
    model = LanguageModel(model_config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_to_run, eta_min=0.0)

    # 5. Training
    model.train()
    pbar = tqdm(train_loader, total=steps_to_run, desc=run_name)
    best_val_acc = 0.0

    for step, (input_ids, labels, _) in enumerate(pbar):
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
    
    # 6. POST-TRAINING ANALYSIS
    if args.compute_inner_loss and args.config == 'palimpsa':
        analyze_palimpsa_reconstruction(model, test_loader, args)
    elif args.compute_inner_loss:
        print(f"Warning: Inner loss analysis not implemented for config '{args.config}'. Skipping.")

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
    
    for seq_len in args.seq_len:
        run_training(args, seq_len)