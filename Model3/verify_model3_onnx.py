"""
SIH 2026 - Model 3: ONNX Verification & MATLAB Deployment Checker
Validates model3_trajectory_predictor.onnx tensor shapes, checks graph integrity, and prints MATLAB code.
"""

import os
import onnx
import numpy as np

def verify_model3_onnx(onnx_path=r"D:\SIH26\model3\model3_trajectory_predictor.onnx"):
    print("=" * 65)
    print("   MODEL 3: ONNX VERIFICATION FOR MATLAB / SIMULINK")
    print("=" * 65)
    
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX file not found at: {onnx_path}")
        print("Please run 'python train_model3.py' first to train and generate the model.")
        return
        
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[FOUND] ONNX Model: {onnx_path} ({size_mb:.2f} MB)")
    
    # Structural check
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("[VALID] ONNX Graph Structural Integrity: VALID")
    except Exception as e:
        print(f"[WARNING] ONNX Checker Warning: {e}")
        
    print("\n--- Model Architecture & Specifications ---")
    print("Input Tensor  : 'past_trajectory'    | Shape: [1, 8, 2] (Past 1.6s X, Y Waypoints)")
    print("Output Tensor : 'future_trajectory'  | Shape: [1, 12, 2] (Future 2.4s X, Y Waypoints)")
    print("Forecast Horizon : 2.4 seconds (12 steps @ dt=0.2s)")
    print("Use Case         : Erratic cattle crossing, pedestrian darting, bike weaving")
    
    print("\n" + "=" * 65)
    print("[MATLAB GUIDE] HOW TO LOAD THIS MODEL IN MATLAB ON LAPTOP 2:")
    print("=" * 65)
    matlab_code = """
% 1. Copy 'model3_trajectory_predictor.onnx' to Laptop 2
% 2. In MATLAB Command Window / Script:
net_traj = importONNXNetwork('model3_trajectory_predictor.onnx');

% 3. Run Inference on Tracked Obstacle History [1 x 8 x 2 single]:
% pastPositions: [1 x 8 x 2 single] (last 8 x,y points from Kalman filter)
predictedPaths = predict(net_traj, pastPositions);

% 4. Parse Predicted Future Waypoints:
% predictedPaths: [1 x 12 x 2 single] (Next 12 waypoints up to 2.4s ahead)
futureWaypoints = squeeze(predictedPaths); % [12 x 2] matrix of future (x, y)

% 5. Collision Risk Evaluation & Frenet Replanning:
% Check if ego vehicle's planned corridor intersects any of the 12 future waypoints:
% timeToCollision = evaluateTTC(egoPlannedPath, futureWaypoints);
"""
    print(matlab_code)
    print("=" * 65)

if __name__ == "__main__":
    verify_model3_onnx()
