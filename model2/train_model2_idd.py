"""
SIH 2026 - Model 2 3D LiDAR Object Detector: 50-Epoch Deep Training Pipeline
Optimized for NVIDIA GeForce RTX 2050 (4 GB VRAM)
Features:
- 50 Epochs with Cosine Annealing Learning Rate
- Early Stopping (Patience = 12 epochs)
- Automatic Best Checkpoint Saver (.pth)
- Production ONNX Exporter (.onnx) for MATLAB / Simulink
"""

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from pointpillars_net import PointPillars3DDetector, compute_model2_loss
from idd_dataset_loader import IDDLidarDataset

def train_optimal_model2(
    epochs=50,
    batch_size=8,
    lr=0.0004,
    patience=12,
    num_points=16384,
    zip_path=r"D:\user\Downloads\idd_mm_supplement.zip",
    labels_npz=r"D:\PPD2\model2\idd_pseudo_labels.npz",
    pretrained_pth=r"D:\PPD2\model2\best_model2_lidar.pth",
    save_dir=r"D:\PPD2\model2"
):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("   MODEL 2: 50-EPOCH 3D LIDAR DEEP TRAINING PIPELINE (IDD MULTIMODAL)")
    print(f"   Max Epochs: {epochs} | Batch Size: {batch_size} | Early Stopping Patience: {patience}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"[DEVICE] Hardware Acceleration: {gpu_name} ({device})")

    # 1. Dataset Preparation
    print(f"\n[1/4] Loading IDD Multimodal point clouds directly from zip...")
    full_dataset = IDDLidarDataset(
        zip_path=zip_path,
        labels_npz=labels_npz,
        max_points=num_points,
        augment=False
    )
    total_samples = len(full_dataset)
    val_size = int(0.2 * total_samples)
    train_size = total_samples - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[SPLIT] {train_size} Training frames | {val_size} Validation frames")

    # 2. Network & Backbone Initialization
    print(f"\n[2/4] Initializing PointPillars 3D Network...")
    model = PointPillars3DDetector(in_channels=4, num_proposals=20, num_classes=3).to(device)
    if os.path.exists(pretrained_pth):
        ckpt = torch.load(pretrained_pth, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state)
        print(f"[PRETRAINED] Initialized backbone from: {pretrained_pth}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_pth_path = os.path.join(save_dir, "best_model2_lidar_idd.pth")
    epochs_no_improve = 0

    # 3. Training Loop
    print(f"\n[3/4] Starting 50-Epoch Deep Optimization Loop...")
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
        n_train = len(train_loader)

        # Validation Phase
        model.eval()
        val_loss, val_box, val_cls, val_conf = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for v_points, v_boxes in val_loader:
                v_points = v_points.to(device)
                v_boxes = v_boxes.to(device)
                v_pred = model(v_points)
                v_l, vb_l, vc_l, vcf_l = compute_model2_loss(v_pred, v_boxes)

                val_loss += v_l.item()
                val_box += vb_l
                val_cls += vc_l
                val_conf += vcf_l

        n_val = len(val_loader)
        avg_v_loss = val_loss / n_val
        avg_v_box = val_box / n_val
        avg_v_conf = val_conf / n_val
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s, lr={current_lr:.6f}) | "
              f"Train Loss: {train_loss/n_train:.4f} (Box:{train_box/n_train:.3f}) | "
              f"Val Loss: {avg_v_loss:.4f} (Box:{avg_v_box:.3f}, Conf:{avg_v_conf:.3f})")

        # Checkpoint Saver & Early Stopping Check
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'num_points': num_points,
                'num_proposals': 20,
                'num_classes': 3,
                'dataset': 'IDD Multimodal'
            }, best_pth_path)
            print(f"   >>> [*] Checkpoint Saved: {best_pth_path} (Best Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[EARLY STOPPING] Validation loss did not improve for {patience} epochs. Stopping.")
                break

    total_duration = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f"[COMPLETE] Model 2 Training Finished in {total_duration/60:.2f} minutes.")
    print(f"[BEST LOSS] Absolute Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 70)

    # 4. Automatic ONNX Production Export
    print(f"\n[4/4] Exporting Production ONNX Model for MATLAB / Simulink...")
    best_ckpt = torch.load(best_pth_path, map_location='cpu', weights_only=False)
    export_model = PointPillars3DDetector(in_channels=4, num_proposals=20, num_classes=3)
    export_model.load_state_dict(best_ckpt['model_state_dict'])
    export_model.eval()

    dummy_input = torch.randn(1, num_points, 4, dtype=torch.float32)
    onnx_destinations = [
        os.path.join(save_dir, "model2_lidar.onnx"),
        os.path.join(save_dir, "model2_lidar_idd.onnx")
    ]
    for p in onnx_destinations:
        torch.onnx.export(
            export_model, dummy_input, p,
            export_params=True, opset_version=12, do_constant_folding=True,
            input_names=['point_cloud'], output_names=['predicted_3d_boxes']
        )
        print(f"[EXPORT] Saved Production ONNX: {p} ({os.path.getsize(p)/1e6:.2f} MB)")

    return best_pth_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 2 on IDD Multimodal 3D LiDAR")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=0.0004, help="Learning rate (default: 0.0004)")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    args = parser.parse_args()

    train_optimal_model2(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )
