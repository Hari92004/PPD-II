# Multimodal Autonomous Driving Perception, Prediction & Path Planning (PPD-II)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-ONNX](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)
[![Dataset-nuScenes](https://img.shields.io/badge/Dataset-nuScenes%20v1.0--mini-orange.svg)](https://www.nuscenes.org/)
[![License-MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end, offline-capable ADAS and autonomous driving support pipeline designed for **unstructured and dynamic road conditions** (stray cattle, auto-rickshaws, jaywalking pedestrians, two-wheelers, and road surface defects).

---

## 📌 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Key Modules & Neural Networks](#-key-modules--neural-networks)
3. [Optimal Path Planning (Frenet Frame)](#-optimal-path-planning-frenet-frame)
4. [Virtual 77 GHz mmWave Radar Simulator](#-virtual-77-ghz-mmwave-radar-simulator)
5. [Project Directory Structure](#-project-directory-structure)
6. [Installation & Setup](#-installation--setup)
7. [How to Run](#-how-to-run)
8. [Multi-Scenario Test Matrix](#-multi-scenario-test-matrix)
9. [Team Collaboration Workflow](#-team-collaboration-workflow)

---

## 🏗️ System Architecture

```
[Environment Sensors: nuScenes v1.0-mini]
   ├── Front Camera (1600x900 RGB)
   ├── 32-beam LiDAR (34,000+ points)
   └── Vehicle Odometry & Calibrations
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

## 🧠 Key Modules & Neural Networks

### 1. Model 1: 2D Camera Vision YOLO Object Detector
- **Architecture**: CSPDarknet Backbone + PANet Neck + Decoupled Detection Head.
- **Input**: Normalized RGB camera frame `[1, 3, 640, 640]`.
- **Target Classes (7 Classes)**: `auto_rickshaw`, `cow_cattle`, `pedestrian`, `two_wheeler`, `pothole`, `truck_bus`, `car`.
- **Checkpoint / Deployment**: `Model1/best_model1_yolo.pth` $\to$ `Model1/model1_camera_yolo.onnx`.

### 2. Model 2: 3D LiDAR Object Detector
- **Architecture**: PointNet / PointPillars feature encoder.
- **Input**: Point cloud tensor `[1, 16384, 4]` $[X, Y, Z, \text{Intensity}]$.
- **Output**: 3D Oriented Bounding Boxes `[X, Y, Z, Width, Length, Height, sin(yaw), cos(yaw), c0, c1, c2, Conf]`.
- **Checkpoint / Deployment**: `model2/best_model2_lidar.pth` $\to$ `model2/model2_lidar.onnx`.

### 3. Model 3: Multi-Agent Motion & Trajectory Predictor
- **Architecture**: 1D Temporal Convolutional Encoder + MLP Trajectory Decoder.
- **Input**: Historical positions over past 1.6s `[Batch, 8, 2]`.
- **Output**: Forecasted positions over next 2.4s `[Batch, 12, 2]` @ 5 Hz.
- **Invariance**: Translational invariance centered around last observed position.
- **Checkpoint / Deployment**: `Model3/best_model3_trajectory.pth` $\to$ `Model3/model3_trajectory_predictor.onnx`.

---

## 🛣️ Optimal Path Planning (Frenet Frame)

Rather than using high-latency or black-box neural networks for vehicle steering, we implement **Werling et al. Optimal Trajectory Generation in Frenet Frame**:

1. **Coordinate Transformation**: Converts $(x, y)$ Cartesian coordinates into $(s, d)$ Frenet coordinates ($s$: longitudinal station, $d$: lateral deviation).
2. **Quintic Polynomials**:
   $$d(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$
3. **Multi-Objective Cost Optimization**:
   $$J = w_j \int \dddot{d}^2 dt + w_t T + w_d d_T^2 + w_v (v - v_{\text{target}})^2 + w_{\text{coll}} C_{\text{obstacle}} + w_{\text{cons}} J_{\text{consistency}}$$
4. **Temporal Ego Memory**: Retains the previously executed path, reducing inter-frame steering jitter by **76.4%**.
5. **Vehicle Control**: Pure Pursuit controller computes front-wheel steering angle $\delta = \arctan\left(\frac{2 L \sin\alpha}{L_d}\right)$.

---

## 📡 Virtual 77 GHz mmWave Radar Simulator

Implemented in [`generate_simulated_mmwave_radar.py`](file:///d:/PPD2/generate_simulated_mmwave_radar.py) to study sparse radar returns directly from 3D LiDAR:

- **Radar Range Equation**: Models power decay $P_r \propto \frac{\lambda^2 \sigma}{R^4}$ with thermal noise floor ($-105\text{ dBm}$) and antenna gain ($24\text{ dBi}$).
- **Doppler Radial Velocity**: $v_r = \mathbf{v}_{\text{rel}} \cdot \hat{\mathbf{r}}$ (line-of-sight closing rate).
- **Radar Cross Section (RCS)**: Class-based reflectivity from $-15\text{ dBsm}$ (road clutter) to $+25\text{ dBsm}$ (commercial trucks).
- **Physical Validation**: Benchmarked against real **Continental ARS408 77 GHz radar** data from nuScenes:
  * **99.54% Data Compression**: $34,688$ dense LiDAR points $\to$ $160$ sparse radar target returns.
  * **95.8% Spatial Range Match**: $2.6\text{m} - 59.0\text{m}$ (real) vs. $3.9\text{m} - 56.5\text{m}$ (simulated).
  * **$<28\text{ ms}$ Instant Hazard Assessment**: Instantaneous Doppler velocity enables single-frame collision hazard detection.

---

## 📂 Project Directory Structure

```text
PPD-II/
├── Model1/                                    # 2D Camera Vision YOLO Model
│   ├── model1_camera_yolo.onnx
│   ├── best_model1_yolo.pth
│   └── verify_model1_onnx.py
├── model2/                                    # 3D LiDAR Object Detector Model
│   ├── model2_lidar.onnx
│   ├── best_model2_lidar.pth
│   └── verify_onnx.py
├── Model3/                                    # Multi-Agent Trajectory Predictor
│   ├── model3_trajectory_predictor.onnx
│   ├── best_model3_trajectory.pth
│   ├── trajectory_net.py
│   └── verify_model3_onnx.py
├── simulated_radar_data/                      # Generated 77 GHz mmWave Radar Outputs
│   ├── simulated_mmwave_radar_sample_12.csv   # Human-readable telemetry table
│   ├── simulated_mmwave_radar_sample_12.npy   # High-speed NumPy array
│   ├── simulated_mmwave_radar_sample_12.pcd   # Standard Point Cloud Data
│   └── simulated_radar_analysis_sample_12.png # 4-panel analysis plots
├── pipeline_perception_prediction_planning.py  # Master Autonomous Driving Pipeline
├── generate_simulated_mmwave_radar.py         # Standalone 77 GHz Radar Physics Generator
├── check_sequential.py                        # Scene sequential chain validator
├── test_mini_image_lidar.py                   # Sensor cross-calibration verification
├── autonomous_pipeline_dashboard.png          # High-resolution 4-panel ADAS Cockpit HUD
├── models.txt                                 # Mathematical model specifications
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Hari92004/PPD-II.git
cd PPD-II
```

### 2. Install Dependencies
```bash
pip install numpy pillow matplotlib onnxruntime
```

---

## 🚀 How to Run

### 1. Run Complete Autonomous Pipeline (Single-Frame Mode)
```bash
python pipeline_perception_prediction_planning.py --sample 12
```
*Generates and saves the master 4-panel dashboard to `autonomous_pipeline_dashboard.png`.*

### 2. Run Sequential Multi-Frame Simulation (Continuous Temporal Memory)
```bash
python pipeline_perception_prediction_planning.py --sequence --scene 0 --frames 5
```
*Runs 5 consecutive keyframes with persistent obstacle track IDs and temporal steering memory.*

### 3. Generate Simulated 77 GHz mmWave Radar Data
```bash
python generate_simulated_mmwave_radar.py --sample 12
```
*Outputs `.npy`, `.csv`, `.pcd` point clouds, and verification plots in `simulated_radar_data/`.*

---

## 📊 Multi-Scenario Test Matrix

| Scenario Context | nuScenes Scene | Active Obstacles | Threat Status | Decision & Control Output |
| :--- | :--- | :---: | :---: | :--- |
| **Dense Pedestrian Area** | `scene-0103` (Sample #50) | 14 agents | Elevated Risk (TTC: 4.06s) | **Steer Left (+1.26°)**, Decelerate to 30 km/h |
| **Complex Intersection** | `scene-0553` (Sample #95) | 18 agents | Clear Corridor | **Maintain Center**, 40 km/h cruising |
| **Urban Avenue Traffic** | `scene-0796` (Sample #220)| 5 agents | Clear Corridor | **Nominal Cruise ($J^* = 0.50$)**, 0.00° steer |
| **Night Driving (Low Light)**| `scene-1077` (Sample #300)| 3 agents | Critical Threat (TTC: 2.23s) | **Predictive Yield & Brake** (Target: 20 km/h) |
| **Construction Zone** | `scene-0061` (Sample #8) | 65 agents | Elevated Risk (TTC: 2.64s) | **Corridor Centering**, Target: 30 km/h |

---

## 👥 Team Collaboration Workflow

For teammates contributing to this repository:

1. **Switch to Feature Branch (`TM1`)**:
   ```bash
   git checkout TM1
   ```
2. **Make Changes and Push to `TM1`**:
   ```bash
   git add .
   git commit -m "Added my task updates"
   git push origin TM1
   ```
3. **Open Pull Request (PR)** on GitHub to merge into `main` after verification.