import torch
import argparse
import wandb
import os
import numpy as np
from tqdm import tqdm

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

# --- METRIC HELPERS ---
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

# --- TRAINING LOOP ---
def train(args):
    print(f"--- MQAR Benchmark ---")
    print(f"Config: {args.config} | SeqLen {args.seq_len} | KV {args.num_kv_pairs}")

    if args.config not in MQAR_CONFIGS:
        raise ValueError(f"Config '{args.config}' not found.")
    
    model_config = MQAR_CONFIGS[args.config]
    model_config.max_position_embeddings = args.seq_len
    model_config.vocab_size = args.vocab_size
    model_config.d_model = args.d_model
    
    # 1. Setup Data
    mqar_conf = MQARConfig(
        vocab_size=args.vocab_size,
        input_seq_len=args.seq_len,
        num_examples=args.steps * args.batch_size, 
        num_kv_pairs=args.num_kv_pairs,
        power_a=0.01 
    )
    test_conf = mqar_conf.model_copy()
    test_conf.num_examples = 400 

    data_config = DataConfig(
        train_configs=[mqar_conf],
        test_configs=[test_conf],
        batch_size=args.batch_size,
        cache_dir=os.path.expanduser(args.cache_dir) if args.cache_dir else None
    )

    print("Preparing Data...")
    train_loader, test_loader = prepare_data(data_config)

    # 2. Setup WandB
    if args.use_wandb:
        wandb.init(
            project="Palimpsa_MQAR", 
            config={**args.__dict__, **model_config.__dict__}, 
            name=f"{args.config}_len{args.seq_len}_kv{args.num_kv_pairs}"
        )

    # 3. Init Model & Optimizer
    model = LanguageModel(model_config).cuda()
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Match Zoology: AdamW with 0.1 weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    
    # Match Zoology: Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=0.0
    )

    # 4. Loop
    model.train()
    pbar = tqdm(train_loader, total=args.steps)
    
    for step, (input_ids, labels, slices) in enumerate(pbar):
        if step >= args.steps: break

        input_ids, labels = input_ids.cuda(), labels.cuda()
        
        output = model(input_ids, labels=labels)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # Step the scheduler
        
        # Logging
        if step % 10 == 0:
            train_acc = compute_accuracy(output.logits, labels)
            lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Loss: {loss.item():.4f} | Acc: {train_acc:.2%}")
            if args.use_wandb:
                wandb.log({
                    "train/loss": loss.item(), 
                    "train/accuracy": train_acc, 
                    "train/lr": lr,
                    "step": step
                })

        if step % 100 == 0 and step > 0:
            val_loss, val_acc = evaluate(model, test_loader)
            if args.use_wandb:
                wandb.log({"val/loss": val_loss, "val/accuracy": val_acc, "step": step})

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="palimpsa", choices=["palimpsa", "gla", "gated_deltanet"])
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_kv_pairs", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--cache_dir", type=str, default="./data_cache")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    
    train(args)