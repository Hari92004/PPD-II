# Model 3: Mixture-of-Experts (MoE) GRU Trajectory Predictor

## 1. Overview & Architecture
This module implements a production-grade **Mixture-of-Experts (MoE) Trajectory Predictor** designed for complex, non-lane Indian road traffic conditions (pedestrians darting, cows crossing, autos weaving, bikes overtaking, trucks/buses cruising).

### Model Architecture
- **Input:** Past 8 observed waypoints $(X, Y)$ at 5 Hz (1.6 s horizon) + Agent Class ID (7 classes).
- **Translation Invariance:** Trajectory coordinates are shifted so the last observed waypoint $P_8$ becomes the origin $(0, 0)$.
- **Shared Kinematic Feature MLP:** Computes step-by-step positions and velocities $[x, y, v_x, v_y]$, projected to a 64-dimensional feature vector.
- **Learned Class Embedding:** 7 Indian classes $\to$ 16-dimensional embedding.
- **4 Specialized GRU Experts:**
  - Each expert is a 2-layer GRU Encoder (hidden size 128) + Autoregressive GRU Decoder (12 future steps).
  - Predicts incremental displacement offsets $(\Delta x, \Delta y)$ per step.
- **Soft Gating Network:**
  - Evaluates Class Embedding + Last Velocity + History Summary Context $\to$ Softmax routing weights $g_1, g_2, g_3, g_4$ (sum = 1.0).
  - Fully deterministic and graph-friendly for ONNX and MATLAB deployment.
- **Load-Balancing Loss ($\mathcal{L}_{\text{balance}}$):**
  - Penalizes deviation of mean batch gate usage from uniform to prevent expert collapse.
- **Cumulative Sum Position Recovery:**
  - Reconstructs future trajectory via $P_t = P_8 + \sum_{\tau=1}^t (\Delta x_\tau, \Delta y_\tau)$.

---

## 2. Directory Structure
```
D:\PPD2\Model3_MoE\
├── moe_dataset.py              # 5 Hz nuScenes track extractor + scene-based split + augmentations
├── moe_gru_net.py              # MoE architecture (4 GRU experts, soft gating, load balancing loss)
├── kalman_baseline.py          # Constant-velocity Kalman filter baseline
├── train_moe.py                # Training and comparative evaluation CLI
├── export_onnx_moe.py          # Exports PyTorch weights to production ONNX
├── verify_model3_moe_onnx.py   # Parity check (|PyTorch - ONNX| < 1e-4) & MATLAB guide
├── model3_moe_trajectory.onnx  # Exported production ONNX model
└── README.md                   # This documentation
```

---

## 3. How to Run

### A. Run Constant-Velocity Kalman Filter Baseline
```bash
python D:\PPD2\Model3_MoE\kalman_baseline.py
```

### B. Train Single-GRU Baseline (1 Expert)
```bash
python D:\PPD2\Model3_MoE\train_moe.py --num_experts 1 --epochs 8 --batch_size 64
```

### C. Train 4-Expert MoE Predictor
```bash
python D:\PPD2\Model3_MoE\train_moe.py --num_experts 4 --lambda_balance 0.1 --epochs 10 --batch_size 64
```

### D. Export to Production ONNX
```bash
python D:\PPD2\Model3_MoE\export_onnx_moe.py --ckpt D:\PPD2\Model3_MoE\best_moe_4exp_lambda0.1.pth --output D:\PPD2\Model3_MoE\model3_moe_trajectory.onnx
```

### E. Verify ONNX Graph & Parity
```bash
python D:\PPD2\Model3_MoE\verify_model3_moe_onnx.py
```

---

## 4. The 7 Indian Traffic Classes
1. `auto_rickshaw` (ID: 0)
2. `cow_cattle` (ID: 1)
3. `pedestrian` (ID: 2)
4. `two_wheeler` (ID: 3)
5. `pothole` (ID: 4)
6. `truck_bus` (ID: 5)
7. `car` (ID: 6)

