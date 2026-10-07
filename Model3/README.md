# Model 3: Multi-Agent Trajectory Predictor

This folder contains the complete training pipeline and ONNX exporter for **Model 3: Multi-Agent Motion & Trajectory Predictor** for unstructured Indian roads.

---

## 📁 Folder Contents

- **`trajectory_dataset.py`**: Synthesizes realistic Indian non-lane traffic motion (erratic cattle crossing, pedestrian darting, bike weaving, vehicle turning).
- **`trajectory_net.py`**: Temporal Convolutional Feature Extractor & Future Trajectory Decoder.
- **`train_model3.py`**: PyTorch training loop optimizing Average Displacement Error (ADE) & Final Displacement Error (FDE) and exporting to ONNX.
- **`verify_model3_onnx.py`**: ONNX graph integrity validator and MATLAB code generator.

---

## ⚡ How to Train Model 3

```bash
cd D:\SIH26\model3
python train_model3.py --epochs 25 --batch_size 64
```

---

## 📦 Deliverables Produced

1. **`best_model3_trajectory.pth`**: Trained PyTorch weights.
2. **`model3_trajectory_predictor.onnx`**: **Copy this file to Laptop 2 for MATLAB!**
3. **`sample_past_trajectory.npy`**: Sample past history waypoints for testing.

---

## 🚀 MATLAB Usage on Laptop 2

```matlab
net_traj = importONNXNetwork('model3_trajectory_predictor.onnx');
futureWaypoints = predict(net_traj, pastHistoryWaypoints);
```
