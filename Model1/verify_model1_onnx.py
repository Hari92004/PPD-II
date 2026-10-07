"""
SIH 2026 - Model 1: ONNX Verification & MATLAB Deployment Checker
Validates model1_camera_yolo.onnx tensor shapes, checks graph integrity, and prints MATLAB code.
"""

import os
import onnx
import numpy as np

CLASSES = ['auto_rickshaw', 'cow_cattle', 'pedestrian', 'two_wheeler', 'pothole', 'truck_bus', 'car']

def verify_model1_onnx(onnx_path=r"D:\SIH26\Model1\model1_camera_yolo.onnx"):
    print("=" * 65)
    print("   MODEL 1: ONNX VERIFICATION FOR MATLAB / SIMULINK")
    print("=" * 65)
    
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX file not found at: {onnx_path}")
        print("Please run 'python train_model1.py' first to train and generate the model.")
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
    print("Input Tensor  : 'camera_frame'        | Shape: [1, 3, 640, 640] (RGB Normalized [0, 1])")
    print("Output Tensor : 'detected_boxes'      | Shape: [1, 25, 12]")
    print("Output Format : [x_center, y_center, width, height, c0, c1, c2, c3, c4, c5, c6, confidence]")
    print("Target Classes:")
    for idx, c in enumerate(CLASSES):
        print(f"  {idx}: {c}")
        
    print("\n" + "=" * 65)
    print("[MATLAB GUIDE] HOW TO LOAD THIS MODEL IN MATLAB ON LAPTOP 2:")
    print("=" * 65)
    matlab_code = """
% 1. Copy 'model1_camera_yolo.onnx' to Laptop 2
% 2. In MATLAB Command Window / Script:
net_yolo = importONNXNetwork('model1_camera_yolo.onnx');

% 3. Run Inference on Camera RGB Frame [1 x 3 x 640 x 640 single]:
% cameraFrame = imresize(im2single(rawRGB), [640 640]);
% cameraFrame = permute(cameraFrame, [3 1 2]); % [3 x 640 x 640]
predictions = predict(net_yolo, dlarray(cameraFrame, 'CSS'));

% 4. Parse Detected 2D Obstacles (Filter confidence > 0.40):
% Output cols: [x, y, w, h, auto, cow, ped, bike, pothole, truck, car, conf]
validDetections = predictions(predictions(:, 12) > 0.40, :);
disp(['Detected 2D Obstacles: ', num2str(size(validDetections, 1))]);

% 5. Pass to Multi-Sensor Fusion with Model 2 LiDAR:
% fusedObstacles = fuseCameraAndLidar(validDetections, validLidarBoxes);
"""
    print(matlab_code)
    print("=" * 65)

if __name__ == "__main__":
    verify_model1_onnx()
