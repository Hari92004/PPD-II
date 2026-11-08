"""
====================================================================================================
Model 3 Training & Evaluation CLI for Mixture-of-Experts (MoE) Trajectory Predictor
====================================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

# Ensure immediate unbuffered console logging
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from moe_dataset import CLASSES, CLASS_TO_IDX, build_moe_datasets
from moe_gru_net import MoEGRUTrajectoryNet, compute_moe_loss

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model, dataloader, device, num_experts):
    model.eval()
    per_class_errors = defaultdict(lambda: {'ade': [], 'fde': []})
    per_class_gates = defaultdict(list)
    all_ade = []
    all_fde = []

    with torch.no_grad():
        for batch in dataloader:
            past = batch['past_pos'].to(device)
            future_gt = batch['future_pos'].to(device)
            class_id = batch['class_id'].to(device)

            pred_future, gates = model(past, class_id)

            diff = pred_future - future_gt # [B, 12, 2]
            step_errors = torch.norm(diff, dim=-1) # [B, 12]
            batch_ade = step_errors.mean(dim=-1).cpu().numpy()
            batch_fde = step_errors[:, -1].cpu().numpy()

            cid_np = class_id.cpu().numpy()
            gates_np = gates.cpu().numpy()

            for i in range(len(cid_np)):
                cname = CLASSES[cid_np[i]]
                ade = float(batch_ade[i])
                fde = float(batch_fde[i])
                all_ade.append(ade)
                all_fde.append(fde)
                per_class_errors[cname]['ade'].append(ade)
                per_class_errors[cname]['fde'].append(fde)
                if num_experts > 1:
                    per_class_gates[cname].append(gates_np[i])

    overall_ade = float(np.mean(all_ade))
    overall_fde = float(np.mean(all_fde))
    overall_miss = float(np.mean(np.array(all_fde) > 2.0) * 100.0)

    per_class_summary = {}
    gate_usage_summary = {}
    for cname in CLASSES:
        if cname in per_class_errors and len(per_class_errors[cname]['ade']) > 0:
            c_ades = per_class_errors[cname]['ade']
            c_fdes = per_class_errors[cname]['fde']
            c_miss = float(np.mean(np.array(c_fdes) > 2.0) * 100.0)
            c_acc = 100.0 - c_miss
            per_class_summary[cname] = {
                'ade': float(np.mean(c_ades)),
                'fde': float(np.mean(c_fdes)),
                'miss_rate': c_miss,
                'accuracy': c_acc,
                'count': len(c_ades)
            }
            if num_experts > 1 and len(per_class_gates[cname]) > 0:
                gate_usage_summary[cname] = np.mean(per_class_gates[cname], axis=0)

    return {
        'ade': overall_ade,
        'fde': overall_fde,
        'miss_rate': overall_miss,
        'accuracy': 100.0 - overall_miss
    }, per_class_summary, gate_usage_summary


def train_moe_pipeline(args):
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 75, flush=True)
    print(f"   TRAINING MODEL 3: MoE-GRU ({args.num_experts} Experts) [HIGH ACCURACY RUN]", flush=True)
    print("=" * 75, flush=True)
    print(f"Configuration: Batch={args.batch_size}, LR={args.lr}, Epochs={args.epochs}, Lambda={args.lambda_balance}, TruckWeight={args.class_weight_truck}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[ACCELERATION] CUDA Enabled ({gpu_name} - {vram:.1f} GB VRAM)", flush=True)
    else:
        print("[INFO] Executing on CPU.", flush=True)

    # Setup per-class loss weights
    class_weights = torch.ones(len(CLASSES), dtype=torch.float32, device=device)
    truck_idx = CLASS_TO_IDX['truck_bus']
    class_weights[truck_idx] = args.class_weight_truck
    print(f"[WEIGHTING] Truck/Bus Loss Weight set to: {args.class_weight_truck}x", flush=True)

    # 1. Dataset Loader
    print("\n[1/4] Loading and preparing Trajectory datasets...", flush=True)
    train_dataset, val_dataset = build_moe_datasets(seed=args.seed, oversample_truck=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Instantiate Model
    print(f"\n[2/4] Initializing Architecture ({args.num_experts} Experts)...", flush=True)
    model = MoEGRUTrajectoryNet(
        num_classes=len(CLASSES),
        class_emb_dim=16,
        hidden_dim=128,
        num_experts=args.num_experts
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}", flush=True)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 4. Training Loop
    print(f"\n[3/4] Commencing Training Loop for {args.epochs} Epochs...", flush=True)
    best_val_fde = float('inf')
    best_truck_acc = 0.0
    best_checkpoint_path = os.path.join(args.save_dir, f"best_moe_{args.num_experts}exp_lambda{args.lambda_balance}.pth")
    patience = args.patience
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        pos_loss_epoch = 0.0
        bal_loss_epoch = 0.0
        t0 = time.time()

        for batch in train_loader:
            past = batch['past_pos'].to(device)
            future = batch['future_pos'].to(device)
            class_id = batch['class_id'].to(device)

            optimizer.zero_grad()
            pred_future, gates = model(past, class_id)

            loss, pos_loss, bal_loss = compute_moe_loss(
                pred_future, future, gates,
                class_id=class_id,
                class_weights=class_weights,
                lambda_balance=args.lambda_balance
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_epoch += loss.item()
            pos_loss_epoch += pos_loss.item()
            bal_loss_epoch += bal_loss.item()

        scheduler.step()
        train_time = time.time() - t0

        avg_loss = total_loss_epoch / len(train_loader)
        avg_pos = pos_loss_epoch / len(train_loader)
        avg_bal = bal_loss_epoch / len(train_loader)

        val_overall, val_per_class, val_gates = evaluate_model(model, val_loader, device, args.num_experts)
        truck_m = val_per_class.get('truck_bus', {'accuracy': 0.0, 'fde': 0.0})

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({train_time:.1f}s) | "
              f"Loss: {avg_loss:.4f} | Val FDE: {val_overall['fde']:.3f}m | Overall Acc: {val_overall['accuracy']:.1f}% | "
              f"Truck Acc: {truck_m['accuracy']:.1f}% (FDE: {truck_m['fde']:.3f}m)", flush=True)

        # Save Best checkpoint when either overall FDE improves or truck accuracy improves significantly
        if val_overall['fde'] < best_val_fde or (truck_m['accuracy'] > best_truck_acc and val_overall['fde'] < best_val_fde * 1.05):
            if val_overall['fde'] < best_val_fde:
                best_val_fde = val_overall['fde']
            if truck_m['accuracy'] > best_truck_acc:
                best_truck_acc = truck_m['accuracy']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_overall,
                'per_class_metrics': val_per_class,
                'num_experts': args.num_experts,
                'lambda_balance': args.lambda_balance
            }, best_checkpoint_path)
            print(f"  --> [NEW BEST SAVED] Val FDE = {best_val_fde:.3f} m | Truck Acc = {truck_m['accuracy']:.1f}%", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] Validation did not improve for {patience} epochs.", flush=True)
                break

    # 5. Final Comprehensive Evaluation
    print("\n" + "=" * 75, flush=True)
    print("   FINAL EVALUATION ON VALIDATION SET (BEST CHECKPOINT)", flush=True)
    print("=" * 75, flush=True)
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    best_overall, best_per_class, best_gates = evaluate_model(model, val_loader, device, args.num_experts)

    print(f"OVERALL RESULTS (Epoch {ckpt.get('epoch')}, 4 Experts):", flush=True)
    print(f"  Overall Accuracy (@ 2.0m) : {best_overall['accuracy']:.2f}%", flush=True)
    print(f"  Overall ADE               : {best_overall['ade']:.3f} m", flush=True)
    print(f"  Overall FDE               : {best_overall['fde']:.3f} m", flush=True)
    print(f"  Miss Rate (>2.0m)         : {best_overall['miss_rate']:.2f}%", flush=True)
    print("-" * 75, flush=True)
    print(f"{'Class':<15} | {'Samples':<8} | {'Accuracy (%)':<14} | {'ADE (m)':<10} | {'FDE (m)':<10}", flush=True)
    print("-" * 75, flush=True)
    for cname in CLASSES:
        if cname in best_per_class:
            m = best_per_class[cname]
            print(f"{cname:<15} | {m['count']:<8} | {m['accuracy']:<14.2f}% | {m['ade']:<10.3f} | {m['fde']:<10.3f}", flush=True)
    print("=" * 75, flush=True)

    return best_overall, best_per_class, best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate MoE GRU Trajectory Predictor")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--lambda_balance", type=float, default=0.1, help="Load balancing loss weight")
    parser.add_argument("--class_weight_truck", type=float, default=2.2, help="Loss weight multiplier for truck_bus class")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default=r"D:\PPD2\Model3_MoE", help="Directory to save checkpoints")
    args = parser.parse_args()

    train_moe_pipeline(args)
