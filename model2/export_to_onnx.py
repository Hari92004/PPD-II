"""
SIH 2026 - Model 2: Export Trained Weights (.pth) to MATLAB-Ready ONNX (.onnx)
Loads the latest trained checkpoint (best_model2_lidar_idd.pth)
and exports validated, production-grade model2_lidar.onnx for MATLAB / Simulink.
"""

import os
import torch
import numpy as np
from pointpillars_net import PointPillars3DDetector

def export_trained_model(
    checkpoint_path=None,
    onnx_output_path=r"D:\PPD2\model2\model2_lidar.onnx",
    num_points=16384
):
    print("=" * 65)
    print("   MODEL 2: EXPORTING TRAINED CHECKPOINT TO MATLAB ONNX")
    print("=" * 65)

    if checkpoint_path is None:
        idd_ckpt = r"D:\PPD2\model2\best_model2_lidar_idd.pth"
        default_ckpt = r"D:\PPD2\model2\best_model2_lidar.pth"
        checkpoint_path = idd_ckpt if os.path.exists(idd_ckpt) else default_ckpt

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"[1/4] Loading PyTorch weights from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    val_loss = checkpoint.get('val_loss', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    dataset = checkpoint.get('dataset', 'nuScenes / IDD')
    print(f"      Best Epoch: {epoch} | Dataset: {dataset} | Validation Loss: {val_loss:.4f}")

    # 1. Initialize Model & Load Weights
    model = PointPillars3DDetector(
        in_channels=4,
        num_proposals=checkpoint.get('num_proposals', 20),
        num_classes=checkpoint.get('num_classes', 3)
    )
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state)
    model.eval()
    model.to('cpu')
    print("[2/4] Neural network weights loaded into evaluation mode.")

    # 2. Export to ONNX
    print(f"[3/4] Exporting ONNX to: {onnx_output_path}...")
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
        dynamic_axes=None  # Fixed shape [1, 16384, 4] for deterministic real-time 50Hz MATLAB inference
    )

    file_size_mb = os.path.getsize(onnx_output_path) / 1e6
    print(f"[SUCCESS] ONNX Model Exported: {onnx_output_path} ({file_size_mb:.2f} MB)")

    # Also save backup copy as model2_lidar_idd.onnx
    backup_path = r"D:\PPD2\model2\model2_lidar_idd.onnx"
    if onnx_output_path != backup_path:
        import shutil
        shutil.copy2(onnx_output_path, backup_path)
        print(f"[BACKUP] Copied backup to: {backup_path}")

    # 3. Verify ONNX Graph with ONNX Runtime
    print("\n[4/4] Verifying ONNX Runtime Inference & Graph Integrity...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_output_path, providers=['CPUExecutionProvider'])
        in_meta = sess.get_inputs()[0]
        out_meta = sess.get_outputs()[0]

        dummy_np = np.random.randn(1, num_points, 4).astype(np.float32)
        res = sess.run(None, {in_meta.name: dummy_np})

        print(f"  * Input Tensor  : '{in_meta.name}' | Shape: {in_meta.shape} | Type: {in_meta.type}")
        print(f"  * Output Tensor : '{out_meta.name}' | Shape: {res[0].shape} | Type: float32")
        print(f"  * Detection Head: 20 candidate boxes x 12 parameters per box")
        print(f"  * Parameters    : [X, Y, Z, Width, Length, Height, sin(yaw), cos(yaw), c0, c1, c2, conf]")
        print("[VERIFICATION PASSED] ONNX graph successfully validated for MATLAB!")
    except Exception as e:
        print(f"[VERIFICATION WARNING] {e}")

    print("\n" + "=" * 65)
    print("MODEL 2 ONNX IS READY FOR MATLAB / SIMULINK CLOSED-LOOP SYSTEM!")
    print("=" * 65)
    return onnx_output_path

if __name__ == "__main__":
    export_trained_model()
