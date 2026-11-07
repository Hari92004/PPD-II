"""
SIH 2026 - Model 2: Export Trained Weights (.pth) to MATLAB-Ready ONNX (.onnx)
Loads best_model2_lidar.pth and creates model2_lidar.onnx without retraining.
"""

import os
import torch
import numpy as np
from pointpillars_net import PointPillars3DDetector
from dataset_loader import NuScenesLidarDataset

def export_trained_model(
    checkpoint_path=r"D:\SIH26\model2\best_model2_lidar.pth",
    onnx_output_path=r"D:\SIH26\model2\model2_lidar.onnx",
    num_points=16384
):
    print("=" * 65)
    print("   MODEL 2: EXPORTING TRAINED CHECKPOINT TO MATLAB ONNX")
    print("=" * 65)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    print(f"[1/3] Loading PyTorch weights from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    val_loss = checkpoint.get('val_loss', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"      Best Epoch: {epoch} | Validation Loss: {val_loss:.4f}")
    
    # Initialize Model & Load Weights
    model = PointPillars3DDetector(
        in_channels=4,
        num_proposals=checkpoint.get('num_proposals', 20),
        num_classes=checkpoint.get('num_classes', 3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to('cpu')
    print("[2/3] Neural network weights loaded into evaluation mode.")
    
    # Export to ONNX
    print(f"[3/3] Exporting ONNX to: {onnx_output_path}...")
    dummy_input = torch.randn(1, num_points, 4, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=12,  # Fully supported by MATLAB 2022a - 2026b Deep Learning Toolbox
        do_constant_folding=True,
        input_names=['point_cloud'],
        output_names=['predicted_3d_boxes'],
        dynamic_axes=None  # Deterministic shape [1, 16384, 4] for real-time 50Hz MATLAB inference
    )
    
    file_size_mb = os.path.getsize(onnx_output_path) / 1e6
    print(f"[SUCCESS] ONNX Model Exported: {onnx_output_path} ({file_size_mb:.2f} MB)")
    
    # Also save a real sample pointcloud from dataset for testing in MATLAB
    print("\nExtracting sample point cloud for MATLAB verification...")
    try:
        dataset = NuScenesLidarDataset()
        sample_pts, _ = dataset[0]
        sample_path = r"D:\SIH26\model2\sample_pointcloud.npy"
        np.save(sample_path, sample_pts.numpy())
        print(f"[SUCCESS] Sample point cloud saved to: {sample_path}")
    except Exception as e:
        print(f"[WARNING] Could not save sample point cloud: {e}")
        
    print("\n" + "=" * 65)
    print("MODEL 2 READY FOR LAPTOP 2!")
    print("=" * 65)

if __name__ == "__main__":
    export_trained_model()
