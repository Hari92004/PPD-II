"""
Train Model 3 (Trajectory Predictor) on Real-World Indian Driving Dataset (IDD Multimodal)
Source: D:\\user\\Downloads\\idd_mm_primary.zip
Extracts real GPS/IMU localization trajectories across drives (d0, d1, d2) at 5 Hz (dt=0.2s).
Trained on NVIDIA GeForce RTX 2050 GPU (Strict GPU Mode).
"""

import os
import sys
import time
import zipfile
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trajectory_net import IndianTrajectoryPredictorNet, compute_trajectory_loss

ZIP_PATH = r"D:\user\Downloads\idd_mm_primary.zip"
SAVE_DIR = r"D:\PPD2\Model3"

class IDDTrajectoryDataset(Dataset):
    """
    Parses real Indian driving GPS trajectories from IDD Multimodal dataset.
    Converts (lat, lon) to metric Cartesian coordinates (meters).
    Generates sliding window sequences: Past 8 steps (1.6s) -> Future 12 steps (2.4s).
    """
    def __init__(self, zip_path, split="train", past_len=8, pred_len=12, stride=1):
        self.past_len = past_len
        self.pred_len = pred_len
        self.seq_len = past_len + pred_len
        
        all_sequences = []
        drives = ['d0', 'd1', 'd2']
        R = 6371000.0 # Earth radius in meters
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            for d in drives:
                csv_name = f'idd_multimodal/primary/{d}/{split}.csv'
                try:
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f)
                        if 'latitude' in df.columns and 'longitude' in df.columns:
                            lat = np.radians(df['latitude'].values)
                            lon = np.radians(df['longitude'].values)
                            lat0, lon0 = lat[0], lon[0]
                            # Equirectangular metric projection
                            x = R * (lon - lon0) * np.cos(lat0)
                            y = R * (lat - lat0)
                            
                            # Subsample from 15Hz to 5Hz (dt=0.2s)
                            x_5hz = x[::3]
                            y_5hz = y[::3]
                            xy_traj = np.stack([x_5hz, y_5hz], axis=-1) # [N, 2]
                            
                            # Sliding window
                            for i in range(0, len(xy_traj) - self.seq_len + 1, stride):
                                seq = xy_traj[i : i + self.seq_len] # [20, 2]
                                all_sequences.append(seq)
                except Exception as e:
                    print(f"  [WARN] Failed to read {csv_name}: {e}")
                    
        self.sequences = np.array(all_sequences, dtype=np.float32)
        print(f"[{split.upper()}] Loaded {len(self.sequences)} real-world Indian vehicle motion sequences.")
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx] # [20, 2]
        past = seq[:self.past_len] # [8, 2]
        future = seq[self.past_len:] # [12, 2]
        return torch.from_numpy(past), torch.from_numpy(future)


def train_idd_model3(epochs=25, batch_size=32, lr=0.002):
    print("=" * 70)
    print("   MODEL 3: REAL-WORLD INDIAN TRAJECTORY PREDICTOR TRAINING")
    print("   Dataset: IDD Multimodal (idd_mm_primary.zip)")
    print("=" * 70)
    
    # 1. Device Setup (Strict GPU Mode)
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available! Strict GPU Mode requires NVIDIA RTX 2050 GPU.")
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[ACCELERATION] CUDA Enabled ({gpu_name} - {vram_gb:.1f} GB VRAM) - STRICT GPU MODE")
    
    # 2. Dataset Loaders
    print("\n[1/4] Extracting & Vectorizing Indian GPS Trajectories...")
    train_dataset = IDDTrajectoryDataset(ZIP_PATH, split="train")
    val_dataset = IDDTrajectoryDataset(ZIP_PATH, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model & Optimizer
    print("\n[2/4] Initializing Temporal Convolutional Predictor Network...")
    model = IndianTrajectoryPredictorNet(past_len=8, pred_len=12, hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_ade = float('inf')
    best_pth_path = os.path.join(SAVE_DIR, "best_model3_idd_trajectory.pth")
    
    # 4. Training Loop
    print(f"\n[3/4] Starting Training ({epochs} Epochs, Batch Size={batch_size})...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
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
        avg_train_ade = train_ade / n_b
        
        # Validation
        model.eval()
        val_ade, val_fde = 0.0, 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                vpred = model(vx)
                _, vade, vfde = compute_trajectory_loss(vpred, vy)
                val_ade += vade
                val_fde += vfde
                
        n_vb = len(val_loader)
        avg_val_ade = val_ade / n_vb
        avg_val_fde = val_fde / n_vb
        epoch_dt = time.time() - t0
        
        is_best = avg_val_ade < best_val_ade
        marker = " [BEST]" if is_best else ""
        if is_best:
            best_val_ade = avg_val_ade
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ade': avg_val_ade,
                'val_fde': avg_val_fde,
                'dataset': 'IDD Multimodal GPS',
                'past_len': 8,
                'pred_len': 12
            }, best_pth_path)
            
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train ADE: {avg_train_ade:.4f}m | Val ADE: {avg_val_ade:.4f}m (FDE: {avg_val_fde:.4f}m) | Time: {epoch_dt:.1f}s{marker}", flush=True)
        
    total_time = time.time() - start_time
    print(f"\n[DONE] IDD Model 3 Training Completed in {total_time/60:.2f} minutes!")
    print(f"[BENCHMARK] Final Best Val ADE: {best_val_ade:.4f} meters")
    print(f"[SAVED] Best PyTorch Weights: {best_pth_path}")
    
    # 5. Export to ONNX
    print("\n[4/4] Exporting Model 3 to Production ONNX...")
    checkpoint = torch.load(best_pth_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to('cpu')
    
    dummy_input = torch.randn(1, 8, 2, dtype=torch.float32)
    onnx_path = os.path.join(SAVE_DIR, "model3_trajectory_predictor.onnx")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['past_trajectory'],
        output_names=['future_trajectory']
    )
    
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[EXPORT] Production ONNX Model Exported: {onnx_path} ({size_mb:.2f} MB)")
    
    return best_val_ade, onnx_path

if __name__ == "__main__":
    train_idd_model3(epochs=25, batch_size=32, lr=0.002)

