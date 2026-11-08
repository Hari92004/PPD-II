"""
====================================================================================================
Model 3: Export MoE-GRU Trajectory Predictor to Production ONNX
====================================================================================================
- Exports trained PyTorch weights to standard ONNX.
- Fixed input shapes:
    past_trajectory : [1, 8, 2]  (float32)
    class_id        : [1]        (int64)
- Fixed output shapes:
    future_trajectory : [1, 12, 2] (float32)
    gating_weights    : [1, 4]      (float32)
- Verifies numerical parity: max absolute difference |PyTorch - ONNX| < 1e-4.
====================================================================================================
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from moe_gru_net import MoEGRUTrajectoryNet

class MoEExportWrapper(nn.Module):
    """
    Export wrapper ensuring exact ONNX tensor naming and deterministic graph.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, past_trajectory, class_id):
        future_pos, gates = self.model(past_trajectory, class_id)
        return future_pos, gates


def export_moe_to_onnx(ckpt_path, output_onnx_path, num_experts=4):
    print("=" * 70)
    print("   EXPORTING MODEL 3 MoE-GRU TO PRODUCTION ONNX")
    print("=" * 70)
    print(f"Loading checkpoint: {ckpt_path}")
    
    device = torch.device('cpu') # Export on CPU for clean platform-independent graph
    model = MoEGRUTrajectoryNet(num_experts=num_experts)
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("[LOADED] PyTorch weights successfully loaded.")
    else:
        print(f"[WARNING] Checkpoint not found at {ckpt_path}. Using initialized weights for export test.")

    model.eval()
    wrapper = MoEExportWrapper(model)

    # Dummy inputs with fixed shapes
    dummy_past = torch.randn(1, 8, 2, dtype=torch.float32)
    dummy_cls = torch.tensor([6], dtype=torch.long) # default 'car'

    # Compute PyTorch reference output
    with torch.no_grad():
        ref_future, ref_gates = wrapper(dummy_past, dummy_cls)
    ref_future_np = ref_future.numpy()
    ref_gates_np = ref_gates.numpy()

    # Export to ONNX
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    input_names = ['past_trajectory', 'class_id']
    output_names = ['future_trajectory', 'gating_weights']

    print(f"Exporting to: {output_onnx_path}...")
    torch.onnx.export(
        wrapper,
        (dummy_past, dummy_cls),
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names
    )
    print(f"[SUCCESS] ONNX file exported ({os.path.getsize(output_onnx_path) / 1e6:.2f} MB)")

    # 1. Structural check via ONNX
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[PASS] ONNX Graph Structural Integrity verified.")

    # 2. Numerical Parity Check via ONNX Runtime
    sess = ort.InferenceSession(output_onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {
        'past_trajectory': dummy_past.numpy(),
        'class_id': dummy_cls.numpy()
    }
    ort_outputs = sess.run(None, ort_inputs)
    onnx_future = ort_outputs[0]
    onnx_gates = ort_outputs[1]

    # Calculate max absolute difference
    diff_future = np.max(np.abs(ref_future_np - onnx_future))
    diff_gates = np.max(np.abs(ref_gates_np - onnx_gates))

    print("\n--- Numerical Parity Verification ---")
    print(f"Future Trajectory Max Abs Diff : {diff_future:.2e}  (Threshold < 1e-4)")
    print(f"Gating Weights    Max Abs Diff : {diff_gates:.2e}  (Threshold < 1e-4)")

    if diff_future < 1e-4 and diff_gates < 1e-4:
        print("[VERIFIED] PyTorch vs ONNX Parity: PASS (< 1e-4)!")
    else:
        print("[FAIL] Output difference exceeds threshold!")

    print("\n--- Model Specifications ---")
    print("Input 1 : 'past_trajectory'   | Shape: [1, 8, 2]  (Past 1.6s @ 5Hz)")
    print("Input 2 : 'class_id'          | Shape: [1]        (0..6 integer)")
    print("Output 1: 'future_trajectory' | Shape: [1, 12, 2] (Future 2.4s @ 5Hz)")
    print("Output 2: 'gating_weights'    | Shape: [1, 4]     (Soft routing probabilities)")
    print("=" * 70)
    return output_onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=r"D:\PPD2\Model3_MoE\best_moe_4exp_lambda0.1.pth")
    parser.add_argument("--output", type=str, default=r"D:\PPD2\Model3_MoE\model3_moe_trajectory.onnx")
    parser.add_argument("--num_experts", type=int, default=4)
    args = parser.parse_args()

    export_moe_to_onnx(args.ckpt, args.output, num_experts=args.num_experts)

