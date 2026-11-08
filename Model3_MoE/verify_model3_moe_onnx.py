"""
====================================================================================================
Model 3: ONNX Verification & MATLAB Deployment Checker for MoE-GRU Trajectory Predictor
====================================================================================================
Validates model3_moe_trajectory.onnx tensor shapes, graph integrity, benchmark latency, and prints MATLAB instructions.
====================================================================================================
"""

import os
import time
import onnx
import onnxruntime as ort
import numpy as np

def verify_model3_moe_onnx(onnx_path=r"D:\PPD2\Model3_MoE\model3_moe_trajectory.onnx"):
    print("=" * 70)
    print("   MODEL 3 MoE-GRU: ONNX VERIFICATION FOR MATLAB & DEPLOYMENT")
    print("=" * 70)

    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found at: {onnx_path}")
        return

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[FOUND] Model File: {onnx_path} ({size_mb:.2f} MB)")

    # 1. Structural Validation
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("[PASS] ONNX Graph Structural Integrity: VALID")

    # 2. Tensor Specs
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print("\n--- Tensor Specifications ---")
    for inp in sess.get_inputs():
        print(f"Input  : '{inp.name}' | Shape: {inp.shape} | Dtype: {inp.type}")
    for out in sess.get_outputs():
        print(f"Output : '{out.name}' | Shape: {out.shape} | Dtype: {out.type}")

    # 3. Latency Benchmark
    past_sample = np.random.randn(1, 8, 2).astype(np.float32)
    cls_sample = np.array([6], dtype=np.int64) # car

    # Warmup
    for _ in range(10):
        _ = sess.run(None, {'past_trajectory': past_sample, 'class_id': cls_sample})

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = sess.run(None, {'past_trajectory': past_sample, 'class_id': cls_sample})
        times.append((time.perf_counter() - t0) * 1000)

    mean_lat = np.mean(times)
    fps = 1000.0 / mean_lat
    print(f"\nInference Latency (CPU, 100 runs): {mean_lat:.2f} ms")
    print(f"Throughput                       : {fps:.1f} FPS")

    # 4. MATLAB Code Instructions
    print("\n" + "=" * 70)
    print("[MATLAB GUIDE] HOW TO IMPORT IN MATLAB / SIMULINK:")
    print("=" * 70)
    matlab_code = """
% 1. Load the MoE Trajectory ONNX Network in MATLAB:
net_moe = importONNXNetwork('model3_moe_trajectory.onnx');

% 2. Prepare Inputs [1 x 8 x 2 single] and [1 x 1 int64]:
% pastPositions: past 1.6s track (8 waypoints)
% classID: 0:auto, 1:cattle, 2:pedestrian, 3:two_wheeler, 4:pothole, 5:truck_bus, 6:car
[futureTrajectory, gatingWeights] = predict(net_moe, pastPositions, classID);

% 3. Extract Future Predicted 12 Waypoints (2.4s horizon at 5 Hz):
% futureTrajectory: [1 x 12 x 2 single]
"""
    print(matlab_code.strip())
    print("=" * 70)

if __name__ == "__main__":
    verify_model3_moe_onnx()

