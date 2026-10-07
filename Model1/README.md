# Model 1: 2D Camera Vision YOLO Object Detector

This folder contains the complete training suite for **Model 1: 2D Camera Perception Detector** for Indian road traffic classes.

---

## 📁 Folder Contents

- **`prepare_dataset.py`**: Generates and formats Indian road scenes into standard YOLO bounding boxes for 7 classes: `[auto_rickshaw, cow_cattle, pedestrian, two_wheeler, pothole, truck_bus, car]`.
- **`yolo_vision_net.py`**: CSPDarknet Feature Pyramid & Decoupled 2D Bounding Box Detection network.
- **`train_model1.py`**: Complete PyTorch training pipeline with auto-save for best weights and automatic MATLAB ONNX export.
- **`verify_model1_onnx.py`**: ONNX graph verification tool and MATLAB code generator.

---

## ⚡ How to Train Model 1

Run the training pipeline:

```bash
cd D:\SIH26\Model1
python train_model1.py --epochs 15 --batch_size 8
```

---

## 📦 Deliverables Produced

1. **`best_model1_yolo.pth`**: Trained PyTorch weights.
2. **`model1_camera_yolo.onnx`**: **Copy this file to Laptop 2 for MATLAB!**
3. **`sample_camera_frame.npy`**: Test sample camera frame.

---

## 🚀 MATLAB Usage on Laptop 2

```matlab
net_yolo = importONNXNetwork('model1_camera_yolo.onnx');
predictions = predict(net_yolo, cameraFrameData);
```
