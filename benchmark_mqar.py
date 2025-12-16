import torch
import argparse
import wandb
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
# 1. METRICS
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

# =============================================================================
# 2. ANALYSIS & SAVING (No Plotting Here)
# =============================================================================
@torch.no_grad()
def compute_and_save_reconstruction_stats(model, dataloader, args, run_name):
    print(f"\n[Analysis] Computing KV Reconstruction Dynamics (Seq Len: {args.seq_len})...")
    
    try:
        from palimpsa.ops.palimpsa import chunk_palimpsa
    except ImportError:
        print("❌ Error: Could not import 'chunk_palimpsa'. Check palimpsa/ops/palimpsa/__init__.py")
        return

    model.eval()
    
    all_layer_errors = [] 
    num_batches = 20 
    iterator = iter(dataloader)
    
    for _ in tqdm(range(num_batches), desc="Analyzing Batches"):
        try:
            input_ids, _, _ = next(iterator)
        except StopIteration:
            break
        input_ids = input_ids.cuda()

        # Handle model wrapping
        if hasattr(model.backbone, "model"):
            base_model = model.backbone.model
        else:
            base_model = model.backbone

        hidden_states = base_model.embeddings(input_ids)

        batch_layer_errors = [] 
        
        for layer in base_model.layers:
            # [FIX] Use .mixer instead of .attn based on your UnifiedBlock definition
            mixer = layer.mixer 
            normed_states = layer.attn_norm(hidden_states)
            
            # --- Palimpsa Projections ---
            # Verify this is actually Palimpsa before trying to access specific attributes
            if not hasattr(mixer, 'b_rank_proj') and args.config == 'palimpsa':
                 # Skip if running GLA but analysis thinks it's Palimpsa (e.g. mixed layers)
                 hidden_states, _ = layer(hidden_states)
                 continue

            q = mixer.q_proj(normed_states).float()
            k = mixer.k_proj(normed_states).float()
            v = mixer.v_proj(normed_states).float()
            
            if mixer.use_short_conv:
                q_orig = mixer.q_proj(normed_states)
                k_orig = mixer.k_proj(normed_states)
                v_orig = mixer.v_proj(normed_states)
                
                # Note: passing cu_seqlens=None assumes padded batch (standard for analysis)
                q_orig, _ = mixer.q_conv1d(q_orig, None)
                k_orig, _ = mixer.k_conv1d(k_orig, None)
                v_orig, _ = mixer.v_conv1d(v_orig, None)
                
                q = q_orig.float()
                k = k_orig.float()
                v = v_orig.float()
            
            head_k_dim = mixer.head_k_dim
            head_v_dim = mixer.head_v_dim
            
            q = rearrange(q, '... (h d) -> ... h d', d=head_k_dim)
            k = rearrange(k, '... (h d) -> ... h d', d=head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=head_v_dim)
            
            # Beta & Gating
            if mixer.b_rank_proj is not None:
                b_in = mixer.b_rank_proj(normed_states)
                b = mixer.b_proj(b_in).float()
                b = rearrange(b, '... (h d) -> ... h d', d=head_v_dim)
                
                bs = torch.sigmoid(mixer.bs_proj(normed_states).float())
                b = torch.sigmoid(b) * bs.unsqueeze(-1)
                
                v_target = v * b
                
                Ip = torch.exp(mixer.Ip_log.float())
            else:
                # Fallback for metaplasticity=False (SimpleGLA) analysis
                # b=0 effectively, so reconstruction logic needs adjustment or skip
                # Here we just set dummy B to avoid crash, but results might mean standard GLA recall
                b = torch.zeros_like(v)
                v_target = torch.zeros_like(v) 
                Ip = torch.ones(mixer.num_v_heads, device=q.device)

            if hasattr(mixer, 'qk_act') and mixer.qk_act == 'siluL2':
                q = F.normalize(F.silu(q), p=2, dim=-1)
                k = F.normalize(F.silu(k), p=2, dim=-1)
            else:
                q = F.softmax(q, dim=-1)
                k = F.softmax(k, dim=-1)
            
            # --- Get Final State ---
            dt = F.softplus(mixer.dt_proj(normed_states).float() + mixer.dt_bias.float())
            A = mixer.A_log.float().exp()
            
            # Get the FINAL state (Mu_T)
            _, final_mu, _ = chunk_palimpsa(
                q=q, k=k, v=v_target, b=b, gt=dt, g=A, Ip=Ip,
                scale=1.0, output_final_state=True
            )
            
            # --- RECONSTRUCTION ---
            # reconstruction[b, l, h, v] = einsum(state[b, h, v, k], k[b, l, h, k])
            reconstruction = einsum(final_mu, k, "b h v k, b l h k -> b l h v")
            
            diff_sq = (reconstruction - v) ** 2
            
            # Use raw difference for visualization (easier to interpret than beta-weighted)
            # or keep weighted if you prefer: weighted_error = b * diff_sq
            error_val = diff_sq
            
            # Mean over Head and Val dim -> [Batch, Seq]
            token_error = error_val.mean(dim=(-2, -1))
            batch_layer_errors.append(token_error)

            # Manually advance hidden states using the actual layer logic
            hidden_states, _ = layer(hidden_states)
        
        # Mean over Layers -> [Batch, Seq]
        if batch_layer_errors:
            batch_avg_layers = torch.stack(batch_layer_errors).mean(dim=0)
            all_layer_errors.append(batch_avg_layers.cpu())

    # --- SAVE ---
    if not all_layer_errors:
        print("No data collected.")
        return

    errors_tensor = torch.cat(all_layer_errors, dim=0) # [Total_Batch, SeqLen]
    
    mean_error = errors_tensor.mean(dim=0).numpy()
    std_error = errors_tensor.std(dim=0).numpy()
    x_axis = np.arange(len(mean_error))

    # [FIX] Ensure directory exists
    os.makedirs("results_reconstruction", exist_ok=True)
    
    filename = f"results_reconstruction/results_{run_name}.npz"
    np.savez(filename, mean=mean_error, std=std_error, x=x_axis, name=run_name)
    print(f"\n[Analysis] Stats saved to: {filename}")
    
    model.train()
# =============================================================================
# 3. MAIN TRAINING LOOP
# =============================================================================
def run_training(args, current_seq_len):
    if args.batch_size is None:
        if current_seq_len >= 1024: batch_size = 64
        elif current_seq_len >= 512: batch_size = 128
        elif current_seq_len >= 256: batch_size = 256
        else: batch_size = 512
    else:
        batch_size = args.batch_size

    target_examples = 6_400_000
    total_steps = int(target_examples / batch_size)
    steps_to_run = args.steps if args.steps > 0 else total_steps

    print(f"\n=== Run: {args.config} | Len {current_seq_len} | Batch {batch_size} | Steps {steps_to_run} ===")

    if args.config not in MQAR_CONFIGS: raise ValueError(f"Config '{args.config}' not found.")
    model_config = MQAR_CONFIGS[args.config]
    model_config.max_position_embeddings = current_seq_len
    model_config.vocab_size = args.vocab_size
    model_config.d_model = args.d_model
    
    print("Generating/Loading Data...")
    mqar_conf_train = MQARConfig(vocab_size=args.vocab_size, input_seq_len=current_seq_len, num_examples=steps_to_run * batch_size, num_kv_pairs=args.num_kv_pairs, power_a=0.01)
    mqar_conf_test = MQARConfig(vocab_size=args.vocab_size, input_seq_len=current_seq_len, num_examples=3000, num_kv_pairs=args.num_kv_pairs, power_a=0.01)

    data_config = DataConfig(train_configs=[mqar_conf_train], test_configs=[mqar_conf_test], batch_size=batch_size, cache_dir=os.path.expanduser(args.cache_dir) if args.cache_dir else None)
    train_loader, test_loader = prepare_data(data_config)

    run_name = f"{args.config}_L{current_seq_len}_D{args.d_model}_KV{args.num_kv_pairs}"
    if args.use_wandb:
        wandb.init(project="Palimpsa_MQAR_Benchmark", group=args.config, name=run_name, config={"seq_len": current_seq_len, "batch_size": batch_size, "steps": steps_to_run, **args.__dict__}, reinit=True)

    model = LanguageModel(model_config).cuda()
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_to_run, eta_min=0.0)

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

    # --- SAVE RESULTS ---
    if args.compute_inner_loss and args.config == 'palimpsa':
        try:
            compute_and_save_reconstruction_stats(model, test_loader, args, run_name)
        except Exception as e:
            print(f"Analysis failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    if args.use_wandb: wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training Args
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
   
    # Standard Training
    seq_lens = args.seq_len
    for s in seq_lens:
        args.seq_len = s 
        run_training(args, s)