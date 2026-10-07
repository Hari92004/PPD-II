"""
SIH 2026 - Model 3: Trajectory Prediction Training & MATLAB Export Pipeline
Trains on non-linear Indian traffic motion, evaluates ADE/FDE metrics,
and exports production-ready ONNX (.onnx) for MATLAB import.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_dataset import get_trajectory_loaders
from trajectory_net import IndianTrajectoryPredictorNet, compute_trajectory_loss

def train_model3(epochs=25, batch_size=64, lr=0.002,
                 save_dir=r"D:\SIH26\model3"):
                 
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 65)
    print("   MODEL 3: TRAJECTORY PREDICTOR TRAINING PIPELINE")
    print("=" * 65)
    
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[ACCELERATION] CUDA Enabled ({gpu_name} - {vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print("[ACCELERATION] CPU Mode (Multi-threaded i7-14700HX)")
        
    # 2. Data Loaders
    print("\n[1/4] Generating & Loading Trajectory Dataset (Cattle, Bikes, Pedestrians)...")
    train_loader, val_loader = get_trajectory_loaders(batch_size=batch_size)
    print(f"[DATASET] Loaded 3200 Train sequences | 800 Validation sequences")

    # 3. Model & Optimizer
    print("\n[2/4] Initializing Trajectory Neural Network...")
    model = IndianTrajectoryPredictorNet(past_len=8, pred_len=12, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_ade = float('inf')
    best_pth_path = os.path.join(save_dir, "best_model3_trajectory.pth")
    
    # 4. Training Loop
    print(f"\n[3/4] Starting Trajectory Training ({epochs} Epochs, Batch Size={batch_size})...")
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss, train_ade, train_fde = 0.0, 0.0, 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss, ade, fde = compute_trajectory_loss(pred, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_ade += ade
            train_fde += fde
            
        scheduler.step()
        n_b = len(train_loader)
        avg_train_loss = train_loss / n_b
        avg_train_ade = train_ade / n_b
        
        # Validation
        model.eval()
        val_loss, val_ade, val_fde = 0.0, 0.0, 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                vpred = model(vx)
                vl, vade, vfde = compute_trajectory_loss(vpred, vy)
                val_loss += vl.item()
                val_ade += vade
                val_fde += vfde
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ade = val_ade / len(val_loader)
        avg_val_fde = val_fde / len(val_loader)
        epoch_time = time.time() - epoch_start
        
        is_best = avg_val_ade < best_val_ade
        marker = " [BEST]" if is_best else ""
        if is_best:
            best_val_ade = avg_val_ade
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_ade': avg_val_ade,
                'val_fde': avg_val_fde,
                'past_len': 8,
                'pred_len': 12
            }, best_pth_path)
            
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train ADE: {avg_train_ade:.4f}m "
              f"| Val ADE: {avg_val_ade:.4f}m (FDE: {avg_val_fde:.4f}m) "
              f"| Time: {epoch_time:.1f}s{marker}", flush=True)

    total_time = time.time() - start_total_time
    print(f"\n[DONE] Model 3 Training Completed in {total_time/60:.2f} minutes!")
    print(f"[BENCHMARK] Final Best ADE: {best_val_ade:.4f} meters (SIH Target Benchmark < 0.35m)")
    print(f"[SAVED] Best Model Weights Saved: {best_pth_path}")

    # 5. Export to ONNX for MATLAB
    print("\n[4/4] Exporting Model 3 to MATLAB-Compatible ONNX...")
    checkpoint = torch.load(best_pth_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to('cpu')
    
    dummy_past = torch.randn(1, 8, 2, dtype=torch.float32)
    onnx_path = os.path.join(save_dir, "model3_trajectory_predictor.onnx")
    
    torch.onnx.export(
        model,
        dummy_past,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['past_trajectory'],
        output_names=['future_trajectory'],
        dynamic_axes=None
    )
    
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[EXPORT] ONNX Model Exported: {onnx_path} ({size_mb:.2f} MB)")
    
    # Save a reference sample past trajectory for MATLAB testing
    sample_past = torch.tensor([
        [[0.0, 0.0], [0.5, 0.1], [1.0, 0.3], [1.6, 0.5],
         [2.1, 0.8], [2.7, 1.2], [3.3, 1.7], [3.9, 2.3]]
    ], dtype=torch.float32)
    sample_path = os.path.join(save_dir, "sample_past_trajectory.npy")
    np.save(sample_path, sample_past.numpy())
    print(f"[SAVED] Sample Trajectory Saved: {sample_path}")
    
    print("\n" + "=" * 65)
    print("MODEL 3 TRAINING & ONNX EXPORT COMPLETE!")
    print("=" * 65)
    print(f"1. PyTorch Checkpoint : {best_pth_path}")
    print(f"2. MATLAB ONNX Model   : {onnx_path}")
    print(f"3. Sample File         : {sample_path}")
    print("\nNext step: Copy 'model3_trajectory_predictor.onnx' to Laptop 2 for MATLAB import!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 3: Multi-Agent Trajectory Predictor")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    args = parser.parse_args()
    
    train_model3(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
