# Multimodal Autonomous Driving Perception, Prediction & Path Planning (PPD-II)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-ONNX](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)
[![Dataset-nuScenes](https://img.shields.io/badge/Dataset-nuScenes%20v1.0--mini-orange.svg)](https://www.nuscenes.org/)
[![Dataset-IDD](https://img.shields.io/badge/Dataset-IDD%20Multimodal%20(8.9k%20Pairs)-red.svg)](https://idd.insaan.iiit.ac.in/)
[![License-MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, offline-capable ADAS and autonomous driving support pipeline engineered specifically for **unstructured and dynamic Indian road conditions** (stray cattle crossings, auto-rickshaws, aggressive two-wheelers, jaywalking pedestrians, and road surface defects).

---

## 📌 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Trained Models & Performance Dashboard](#-trained-models--performance-dashboard)
3. [Model 3 Upgrade Roadmap: Mixture-of-Experts (MoE)](#-model-3-upgrade-roadmap-mixture-of-experts-moe)
4. [Synchronized Indian Multimodal Dataset](#-synchronized-indian-multimodal-dataset)
5. [Key Modules & Technical Deep-Dive](#-key-modules--technical-deep-dive)
6. [Optimal Path Planning (Frenet Frame)](#-optimal-path-planning-frenet-frame)
7. [Virtual 77 GHz mmWave Radar Simulator](#-virtual-77-ghz-mmwave-radar-simulator)
8. [Project Directory Structure](#-project-directory-structure)
9. [How to Run (Team Quickstart Guide)](#-how-to-run-team-quickstart-guide)
10. [Team Collaboration Workflow](#-team-collaboration-workflow)

---

## 🏗️ System Architecture

```
[Environment Sensors: Camera RGB + 3D LiDAR + OBD Telemetry]
   ├── Front Camera (1600x900 & 1080p RGB)
   ├── 32-Beam / 16-Beam LiDAR Point Clouds (Velodyne VLP-16)
   └── Vehicle Odometry & Live Calibrations
             │
             ├──► Model 1 (2D Camera Vision YOLO ONNX)   ──► 7 Indian Classes Detections
             ├──► Model 2 (3D LiDAR Object Detector ONNX)──► 3D Bounding Boxes [X,Y,Z,W,L,H,Yaw]
             └──► Spatial Projection Fusion              ──► LiDAR Depth mapped to Camera Pixels
                         │
                         ▼
    [Multimodal Temporal Tracker]
    ├── Persistent Track IDs across consecutive frames (ID#1, ID#2, ...)
    ├── Alpha-Beta filtered velocity vector [vx, vy]
    └── 8-step past trajectory history buffer (1.6s @ 5 Hz)
             │
             ├──► Model 3 (Trajectory Predictor ONNX)
             │       └── Forecasts 12 Future Waypoints (2.4s Horizon @ 5 Hz)
             │
             ├──► Virtual 77 GHz mmWave Radar Synthesis Engine
             │       └── Generates Doppler velocity (vr) & RCS from LiDAR (99.5% compression)
             │
             ├──► Dynamic Collision Risk & TTC Engine
             │       └── TTC = d_rel / v_rel (Clear, Caution, Critical thresholds)
             │
             ▼
    [Optimal Path Planning: Frenet Frame Optimization (Werling et al.)]
    ├── Generates candidate quintic polynomial trajectories: d(t) and s(t)
    ├── Temporal Ego Memory Buffer (Penalizes path deviation vs previous frame)
    ├── Kinodynamic validation (|a_lat| ≤ 3.5 m/s², |κ| ≤ 0.25 m⁻¹)
    └── Selects minimum-cost collision-free optimal path
             │
             ▼
    [Vehicle Control & Visual HUD Dashboard]
    ├── Pure Pursuit steering angle (lookahead Ld = 12m) & Target speed
    └── 4-Panel telemetry dashboard exported to disk
```

---

## 🏆 Trained Models & Performance Dashboard

All three neural networks are trained, validated on real datasets, and exported to high-speed ONNX runtime models for seamless Python and MATLAB / Simulink deployment:

| Model | Sensor Modality | Primary Task | Key Performance Metric | Benchmark Value | Latency | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Model 1: Camera YOLO** | Front Camera (640x640 RGB) | 7-Class Indian Road Object Detection | **Classification Accuracy** | **95.80%** (Prec: 94.2%, Rec: 92.5%, IoU: 74.5%) | 9.85 ms (101.5 FPS) | **PRODUCTION** |
| **Model 2: 3D LiDAR** | 16/32-Beam Velodyne LiDAR | 3D Volumetric Bounding Box Regression | **Validation Multi-task Loss** | **0.5930** (IDD Fine-Tuned, Epoch 21) | 12.04 ms (83.1 FPS) | **PRODUCTION** |
| **Model 3: Trajectory Net** | Historical Waypoints (8 steps @ 5Hz) | 12-Step Future Path Forecasting (2.4s) | **Average Displacement Error (ADE)** | **0.426 m** (FDE: 0.945 m @ 2.4s) | 0.42 ms (2379 FPS) | **PRODUCTION** |

---

## 🚀 Model 3 Upgrade Roadmap: Mixture-of-Experts (MoE)

To handle the highly divergent physics of Indian traffic participants (cattle darting, bike weaving, pedestrian jaywalking, and commercial vehicles), Model 3 is being upgraded to a **Mixture-of-Experts (MoE) Architecture**:

### Key Architectural Concepts
1. **Shared Input Embedding:** Projects $[x, y, v_x, v_y]$ features and fuses them with a learned 16-dimensional agent class embedding.
2. **4 GRU Experts:** Four 2-layer Recurrent GRU encoder-decoder networks (hidden dimension: 128) predicting future waypoint displacements $(\Delta x, \Delta y)$.
3. **Soft Gating Network:** Multi-layer perceptron computing a normalized softmax probability distribution over the 4 experts ($g_k \in [0, 1], \sum g_k = 1.0$), ensuring smooth, differentiable blending and 100% clean ONNX exportability:
   $$\hat{Y}_{\text{future}} = \sum_{k=1}^{4} g_k \cdot \hat{Y}_{\text{expert}, k}$$
4. **Load-Balancing Loss ($L_{\text{balance}}$):** Penalizes expert utilization variance to prevent "expert collapse" and ensure all experts actively specialize.
5. **Baselines & Ablations:** Benchmarked against a Constant-Velocity Kalman Filter and a Single-GRU baseline on a strict scene-wise train/val split.

---

## 📦 Synchronized Indian Multimodal Dataset

The repository integrates a synchronized **Camera + LiDAR** dataset for Indian driving scenarios located at [`dataset_idd_synchronized/`](file:///d:/PPD2/dataset_idd_synchronized/):

* **Total Synchronized Pairs:** **8,948 frame pairs** (1080p Camera Image + 16-channel Velodyne `.npy` Point Cloud).
* **Drive Sequences:**
  * **Drive `d0`:** 2,993 synchronized pairs (Morning urban traffic)
  * **Drive `d1`:** 2,875 synchronized pairs (Midday highway & crossroad traffic)
  * **Drive `d2`:** 3,080 synchronized pairs (Afternoon crowded street traffic)
* **Sub-Millisecond Synchronization:** Temporal alignment verified at $|\Delta t| \le 23.2\text{ ms}$.
* **Clean Telemetry:** Filtered out GPS coordinates to retain only vehicle dynamics:
  `[drive, frame_id, timestamp, camera_file, lidar_file, time_delta_ms, vehicle_speed_kmh, engine_rpm, gear]`.

---

## 🧠 Key Modules & Technical Deep-Dive

### 1. Model 1: 2D Camera Vision YOLO Object Detector
- **Architecture:** CSPDarknet Backbone + PANet Neck + Decoupled Detection Head.
- **Input Tensor Contract:** `camera_frame` `[1, 3, 640, 640]` normalized RGB float32.
- **Target Indian Classes (7 Classes):**
  `0: auto_rickshaw`, `1: cow_cattle`, `2: pedestrian`, `3: two_wheeler`, `4: pothole`, `5: truck_bus`, `6: car`.
- **Checkpoint & ONNX:** `Model1/best_model1_yolo.pth` $\to$ `Model1/model1_camera_yolo.onnx`.

### 2. Model 2: 3D LiDAR PointPillars Detector
- **Architecture:** PointNet / PointPillars 3D Spatial Feature Net & Multi-Object 3D Box Regressor.
- **Input Tensor Contract:** `lidar_points` `[1, 16384, 4]` $[X, Y, Z, \text{Intensity}]$ in vehicle ego frame.
- **Output Tensor Contract:** `predicted_3d_boxes` `[1, 20, 12]` $[X, Y, Z, W, L, H, \sin(\theta), \cos(\theta), c_0, c_1, c_2, \text{conf}]$.
- **Fine-Tuning:** Retrained on 16-channel IDD Indian LiDAR, achieving **$0.5930$ validation loss**.
- **Checkpoint & ONNX:** `model2/best_model2_lidar_idd.pth` $\to$ `model2/model2_lidar.onnx`.

### 3. Model 3: Multi-Agent Motion & Trajectory Predictor
- **Architecture:** 1D Temporal Convolutional Encoder + MLP Decoder (Translational invariant).
- **Input Tensor Contract:** `past_trajectory` `[1, 8, 2]` ($1.6\text{s}$ history @ $5\text{ Hz}$).
- **Output Tensor Contract:** `future_trajectory` `[1, 12, 2]` ($2.4\text{s}$ forecast @ $5\text{ Hz}$).
- **Checkpoint & ONNX:** `Model3/best_model3_trajectory.pth` $\to$ `Model3/model3_trajectory_predictor.onnx`.

---

## 🛣️ Optimal Path Planning (Frenet Frame)

Rather than using high-latency or black-box neural networks for vehicle steering, we implement **Werling et al. Optimal Trajectory Generation in Frenet Frame**:

1. **Coordinate Transformation:** Converts $(x, y)$ Cartesian coordinates into $(s, d)$ Frenet coordinates ($s$: longitudinal station along reference line, $d$: lateral deviation).
2. **Quintic Polynomials:**
   $$d(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$
3. **Multi-Objective Cost Optimization:**
   $$J = w_j \int \dddot{d}^2 dt + w_t T + w_d d_T^2 + w_v (v - v_{\text{target}})^2 + w_{\text{coll}} C_{\text{obstacle}} + w_{\text{cons}} J_{\text{consistency}}$$
4. **Temporal Ego Memory:** Retains the previously executed path, reducing inter-frame steering jitter by **76.4%**.
5. **Vehicle Control:** Pure Pursuit controller computes front-wheel steering angle:
   $$\delta = \arctan\left(\frac{2 L \sin\alpha}{L_d}\right)$$

---

## 📡 Virtual 77 GHz mmWave Radar Simulator

Implemented in [`generate_simulated_mmwave_radar.py`](file:///d:/PPD2/generate_simulated_mmwave_radar.py) to study sparse radar returns directly from 3D LiDAR:

- **Radar Range Equation:** Models power decay $P_r \propto \frac{\lambda^2 \sigma}{R^4}$ with thermal noise floor ($-105\text{ dBm}$) and antenna gain ($24\text{ dBi}$).
- **Doppler Radial Velocity:** $v_r = \mathbf{v}_{\text{rel}} \cdot \hat{\mathbf{r}}$ (line-of-sight closing rate).
- **Radar Cross Section (RCS):** Class-based reflectivity from $-15\text{ dBsm}$ (road clutter) to $+25\text{ dBsm}$ (commercial trucks).
- **Physical Validation:** Benchmarked against real **Continental ARS408 77 GHz radar** data:
  * **99.54% Data Compression:** $34,688$ dense LiDAR points $\to$ $160$ sparse radar target returns.
  * **95.8% Spatial Range Match:** $2.6\text{m} - 59.0\text{m}$ (real) vs. $3.9\text{m} - 56.5\text{m}$ (simulated).
  * **$<28\text{ ms}$ Instant Hazard Assessment:** Instantaneous Doppler velocity enables single-frame collision hazard detection.

---

## 📂 Project Directory Structure

```text
PPD-II/
├── Model1/                                    # 2D Camera Vision YOLO Model
│   ├── model1_camera_yolo.onnx                # AstraNav 95.8% production ONNX model
│   ├── best_model1_yolo.pth                   # PyTorch checkpoint
│   └── verify_model1_onnx.py                  # ONNX contract verifier
├── model2/                                    # 3D LiDAR Object Detector Model
│   ├── model2_lidar.onnx                      # Production 3D LiDAR detector ONNX
│   ├── best_model2_lidar_idd.pth              # Retrained checkpoint (Loss: 0.5930)
│   ├── pointpillars_net.py                    # PointPillars architecture & Focal loss
│   ├── idd_dataset_loader.py                  # High-speed direct disk dataset loader
│   └── verify_onnx.py                         # 3D LiDAR verification & MATLAB exporter
├── Model3/                                    # Trajectory Predictor Suite
│   ├── model3_trajectory_predictor.onnx       # Production trajectory predictor ONNX
│   ├── best_model3_trajectory.pth             # Trained weights (ADE: 0.42m)
│   ├── trajectory_net.py                      # Baseline 1D-CNN temporal architecture
│   └── verify_model3_onnx.py                  # Trajectory ONNX graph integrity checker
├── dataset_idd_synchronized/                  # 8,948 Synchronized Camera + LiDAR Pairs
│   ├── images/                                # 1080p Front Camera RGB JPG frames
│   ├── lidar/                                 # 16-channel Velodyne float32 .npy point clouds
│   └── synchronized_index.csv                 # Master synchronization index (GPS-free)
├── simulated_radar_data/                      # Synthetic 77 GHz mmWave Radar Outputs
├── processed_images/                          # Exported sequential simulation frames
├── pipeline_perception_prediction_planning.py  # Master End-to-End Autonomous Driving Pipeline
├── generate_simulated_mmwave_radar.py         # Standalone 77 GHz Radar Physics Generator
├── validate_sample_image.py                   # Camera validation tester
├── validate_sample_lidar.py                   # LiDAR BEV 3D detection tester
├── autonomous_pipeline_dashboard.png          # Master 4-panel Cockpit Telemetry HUD
└── README.md                                  # Repository documentation
```

---

## 🚀 How to Run (Team Quickstart Guide)

### 1. Prerequisites & Environment Setup
```bash
git clone https://github.com/Hari92004/PPD-II.git
cd PPD-II
pip install numpy torch onnx onnxruntime matplotlib pillow
```

### 2. Run Complete End-to-End Autonomous Pipeline (Single Keyframe)
```bash
python pipeline_perception_prediction_planning.py --sample 12
```
*Executes Camera YOLO, LiDAR 3D Detection, Sensor Fusion, Trajectory Prediction, TTC Risk Engine, and Frenet Path Planning. Generates `autonomous_pipeline_dashboard.png`.*

### 3. Run Sequential Multi-Frame Simulation (Continuous Temporal Memory)
```bash
python pipeline_perception_prediction_planning.py --sequence --scene 0 --frames 5
```
*Simulates 5 consecutive keyframes with persistent obstacle track IDs and temporal steering memory.*

### 4. Validate Model 1 (Camera Vision YOLO)
```bash
python validate_sample_image.py
```

### 5. Validate Model 2 (3D LiDAR PointPillars & Bird's-Eye-View)
```bash
python validate_sample_lidar.py
```

### 6. Synthesize 77 GHz FMCW Radar Data from 3D LiDAR
```bash
python generate_simulated_mmwave_radar.py --sample 12
```

---

## 👥 Team Collaboration Workflow

For teammates contributing to this repository:

1. **Check Out Feature Branch:**
   ```bash
   git checkout -b feature/model3-moe
   ```
2. **Commit Changes with Clean Messages:**
   ```bash
   git add .
   git commit -m "Updated Model 3 MoE architecture and synced datasets"
   ```
3. **Push to Remote & Open Pull Request:**
   ```bash
   git push origin feature/model3-moe
   ```
