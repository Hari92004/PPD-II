"""
SIH 2026 - Model 2: 3D LiDAR Object Detector Training Pipeline
Trains on nuScenes v1.0-mini point clouds, saves PyTorch weights (.pth),
and exports production-ready ONNX (.onnx) for MATLAB import.
"""

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset_loader import NuScenesLidarDataset
from pointpillars_net import PointPillars3DDetector, compute_model2_loss

def train_model2(epochs=15, batch_size=8, lr=0.001, num_points=16384,
                 data_root=r"D:\SIH26\dataset mini\v1.0-mini",
                 save_dir=r"D:\SIH26\model2"):
    
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 65)
    print("   MODEL 2: 3D LIDAR OBJECT DETECTOR TRAINING PIPELINE")
    print("=" * 65)
    
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[ACCELERATION] CUDA Enabled ({gpu_name} - {vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print(f"[ACCELERATION] CPU Mode (Multi-threaded)")
        
    # 2. Dataset Preparation
    print(f"\n[1/4] Loading nuScenes dataset from: {data_root}...")
    full_dataset = NuScenesLidarDataset(data_root=data_root, max_points=num_points)
    total_samples = len(full_dataset)
    
    val_size = int(0.2 * total_samples)
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[SPLIT] {train_size} Training samples | {val_size} Validation samples")

    # 3. Model & Optimizer
    print(f"\n[2/4] Initializing PointPillars 3D Network...")
    model = PointPillars3DDetector(in_channels=4, num_proposals=20, num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_pth_path = os.path.join(save_dir, "best_model2_lidar.pth")
    
    # 4. Training Loop
    print(f"\n[3/4] Starting Training ({epochs} Epochs, Batch Size={batch_size})...")
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss, train_box, train_cls, train_conf = 0.0, 0.0, 0.0, 0.0
        
        for batch_points, batch_boxes in train_loader:
            batch_points = batch_points.to(device)
            batch_boxes = batch_boxes.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_points)
            loss, b_l, c_l, cf_l = compute_model2_loss(predictions, batch_boxes)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_box += b_l
            train_cls += c_l
            train_conf += cf_l
            
        scheduler.step()
        n_batches = len(train_loader)
        avg_train_loss = train_loss / n_batches
        avg_box = train_box / n_batches
        avg_cls = train_cls / n_batches
        avg_conf = train_conf / n_batches
        
        # Validation Phase
        model.eval()
        val_loss, val_box = 0.0, 0.0
        with torch.no_grad():
            for v_points, v_boxes in val_loader:
                v_points = v_points.to(device)
                v_boxes = v_boxes.to(device)
                v_pred = model(v_points)
                v_l, vb_l, _, _ = compute_model2_loss(v_pred, v_boxes)
                val_loss += v_l.item()
                val_box += vb_l
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_box = val_box / len(val_loader)
        epoch_duration = time.time() - epoch_start
        
        # Checkpoint Saving
        is_best = avg_val_loss < best_val_loss
        marker = " [BEST]" if is_best else ""
        if is_best:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'num_points': num_points,
                'num_proposals': 20,
                'num_classes': 3
            }, best_pth_path)
            
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {avg_train_loss:.4f} (Box:{avg_box:.3f}, Cls:{avg_cls:.3f}) "
              f"| Val Loss: {avg_val_loss:.4f} (Box:{avg_val_box:.3f}) "
              f"| Time: {epoch_duration:.1f}s{marker}", flush=True)

    total_training_time = time.time() - start_total_time
    print(f"\n[DONE] Training Completed in {total_training_time/60:.2f} minutes!")
    print(f"[SAVED] Best Model Weights Saved: {best_pth_path}")

    # 5. ONNX Export for MATLAB
    print(f"\n[4/4] Exporting Model to MATLAB-Compatible ONNX...")
    checkpoint = torch.load(best_pth_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to('cpu')
    
    dummy_input = torch.randn(1, num_points, 4, dtype=torch.float32)
    onnx_path = os.path.join(save_dir, "model2_lidar.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12, # Fully supported by MATLAB 2022a - 2026b
        do_constant_folding=True,
        input_names=['point_cloud'],
        output_names=['predicted_3d_boxes'],
        dynamic_axes=None # Fixed batch=1 for ultra-fast deterministic MATLAB simulation
    )
    
    file_size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[EXPORT] ONNX Model Exported: {onnx_path} ({file_size_mb:.2f} MB)")
    
    # Save a reference test sample (.npy) for easy validation on Laptop 2
    sample_pts, sample_boxes = full_dataset[0]
    test_sample_path = os.path.join(save_dir, "sample_pointcloud.npy")
    np.save(test_sample_path, sample_pts.numpy())
    print(f"[SAVED] Sample Point Cloud Saved: {test_sample_path}")
    
    print("\n" + "=" * 65)
    print("MODEL 2 TRAINING & ONNX EXPORT COMPLETE!")
    print("=" * 65)
    print(f"1. PyTorch Checkpoint : {best_pth_path}")
    print(f"2. MATLAB ONNX Model   : {onnx_path}")
    print(f"3. Sample Point Cloud  : {test_sample_path}")
    print("\nNext step: Copy 'model2_lidar.onnx' to Laptop 2 for MATLAB import!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 2: 3D LiDAR Object Detector")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--points", type=int, default=16384, help="Number of points per cloud")
    args = parser.parse_args()
    
    train_model2(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_points=args.points)
