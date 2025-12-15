import torch
import argparse
import wandb
import os
import numpy as np
from tqdm import tqdm

# --- IMPORTS ---
try:
    from palimpsa import PalimpsaConfig, PalimpsaForCausalLM
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from palimpsa.models.palimpsa.configuration_palimpsa import PalimpsaConfig
    from palimpsa.models.palimpsa.modeling_palimpsa import PalimpsaForCausalLM

# Import Data Pipeline
from data.data_mqar.config import DataConfig
from data.data_mqar.associative_recall import MQARConfig
from data.data_mqar.utils import prepare_data

# FLA Baselines
try:
    from fla.models import GLAForCausalLM, GLAConfig
    from fla.models import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    print("⚠️ FLA not found. Baselines unavailable.")

# --- METRIC HELPERS ---
def compute_accuracy(logits, labels, ignore_index=-100):
    """
    Computes accuracy while ignoring the ignore_index (usually padding).
    """
    # logits: [batch, seq_len, vocab_size]
    # labels: [batch, seq_len]
    preds = torch.argmax(logits, dim=-1)
    
    mask = labels != ignore_index
    correct = (preds == labels) & mask
    
    # Avoid division by zero
    total_valid = mask.sum().float()
    if total_valid == 0:
        return 0.0
        
    accuracy = correct.sum().float() / total_valid
    return accuracy.item()

@torch.no_grad()
def evaluate(model, dataloader):
    """
    Runs the model on the validation set and returns avg loss and accuracy.
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    steps = 0
    
    for input_ids, labels, slices in dataloader:
        input_ids, labels = input_ids.cuda(), labels.cuda()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        acc = compute_accuracy(outputs.logits, labels)
        
        total_loss += loss.item()
        total_acc += acc
        steps += 1
        
    model.train()
    return total_loss / steps, total_acc / steps

# --- TRAINING LOOP ---
def train(args):
    print(f"--- MQAR Benchmark: {args.model} | SeqLen {args.seq_len} | KV {args.num_kv_pairs} ---")
    
    # 1. Setup Data
    mqar_conf = MQARConfig(
        vocab_size=args.vocab_size,
        input_seq_len=args.seq_len,
        num_examples=args.steps * args.batch_size, 
        num_kv_pairs=args.num_kv_pairs,
        power_a=0.01 
    )
    
    # We create a smaller config for testing to keep valid loop fast
    test_conf = mqar_conf.model_copy()
    test_conf.num_examples = 200 # Small validation set

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
            config=args, 
            name=f"{args.model}_len{args.seq_len}_kv{args.num_kv_pairs}"
        )

    # 3. Setup Model
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
        raise ValueError(f"Model {args.model} not found.")

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. Loop
    model.train()
    pbar = tqdm(train_loader, total=args.steps)
    
    for step, (input_ids, labels, slices) in enumerate(pbar):
        if step >= args.steps: break

        input_ids, labels = input_ids.cuda(), labels.cuda()
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Logging
        if step % 10 == 0:
            # Compute train accuracy on the fly for the current batch
            train_acc = compute_accuracy(outputs.logits, labels)
            
            pbar.set_description(f"Loss: {loss.item():.4f} | Acc: {train_acc:.2%}")
            
            if args.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/accuracy": train_acc,
                    "step": step
                })

        # Validation Loop (every 50 steps)
        if step % 50 == 0 and step > 0:
            val_loss, val_acc = evaluate(model, test_loader)
            if args.use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "step": step
                })

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