# Model 2: 3D LiDAR Object Detector Training & MATLAB Export Suite

This folder is dedicated exclusively for **Laptop 1 (Training Machine)** to train the 3D LiDAR Perception Neural Network on the local **nuScenes v1.0-mini** dataset and produce the production `.onnx` model for **Laptop 2 (MATLAB / Simulink Machine)**.

---

## 📁 Folder Contents

- **`dataset_loader.py`**: Directly reads 32-beam `.pcd.bin` binary point clouds from `D:\SIH26\dataset mini\v1.0-mini`, transforms global annotations to the LiDAR sensor frame, and standardizes coordinates.
- **`pointpillars_net.py`**: Point-based 3D Spatial Feature Net & Multi-Object 3D Bounding Box Regressor.
- **`train_model2.py`**: Full training pipeline with GPU/CUDA acceleration, multi-task loss (Smooth L1 + BCE), checkpoint saver, and automatic MATLAB-compatible ONNX export.
- **`verify_onnx.py`**: Validates the exported ONNX model structure, input/output tensors, and provides MATLAB copy-paste commands.

---

## ⚡ How to Run Training on this Laptop

Open terminal/command prompt in this folder:

```bash
cd D:\SIH26\model2
python train_model2.py --epochs 15 --batch_size 8
```

### Options:
- `--epochs`: Number of training epochs (default: `15`)
- `--batch_size`: Batch size (default: `8`)
- `--points`: Point cloud sample size (default: `16384`)
- `--lr`: Learning rate (default: `0.001`)

---

## 📦 What Files are Produced

When training finishes, you will see:
1. **`best_model2_lidar.pth`**: Saved PyTorch model checkpoint.
2. **`model2_lidar.onnx`**: **The file to copy to Laptop 2!**
3. **`sample_pointcloud.npy`**: A test point cloud for quick validation in MATLAB.

---

## 🚀 Transfer to Laptop 2 (MATLAB)

1. Copy **`model2_lidar.onnx`** to a USB drive or local share.
2. On Laptop 2, open MATLAB and run:
   ```matlab
   net = importONNXNetwork('model2_lidar.onnx');
   ```
3. Pass your simulation LiDAR point clouds directly into `predict(net, ptCloud)` in your Simulink closed-loop system!
