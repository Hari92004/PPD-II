# Autonomous Vehicle Multi-Modal Perception, Prediction, Planning & Cockpit Flow

**Complete Standalone Autonomous Driving Package for Indian Road Driving (IDD Synchronized Dataset)**

---

## 1. System Architecture Flow

```
                                [ Synchronized Indian Driving Dataset (IDD) ]
                                          /                       \
                                         /                         \
                       [ Monocular Camera (Front) ]        [ LiDAR 3D Point Cloud ]
                                    |                                 |
                                    v                                 v
                       [ Model 1: YOLOv8 2D ]           [ Model 2: PointPillars 3D ]
                       (Bounding Boxes: Green)          (1-to-1 Spatial Projection Fusion)
                                    \                                 /
                                     \                               /
                                      v                             v
                                 [ Spatial Camera-LiDAR Calibration Fusion ]
                                        u = (-Y·fx / X) + cx, v = (-Z·fy / X) + cy
                                                    |
                                                    v
                                 [ 77 GHz mmWave FMCW Radar Simulation Scope ]
                                 • Texas Instruments AWR1843 MIMO Radar (77-81 GHz)
                                 • Range: Beat frequency fb, BW = 4 GHz, ΔR = 3.75 cm
                                 • Direct Doppler vr = (x·vx + y·vy) / R
                                 • Polar Scope (0°, ±15°, ±30°, ±45°), RCS Estimation
                                                    |
                                                    v
                              [ Model 3: Mixture-of-Experts (MoE) Trajectory Predictor ]
                              • 4 Kinematically Specialized GRU Experts
                              • Softmax Gating Network (Dynamic Weight Allocation)
                              • Forecasts 12 Future Steps (2.4s Horizon @ 5 Hz)
                                                    |
                                                    v
                             [ 3-Zone Frenet Quintic Path Planner & Evasion ]
                             • Zone 1 (s <= 4.2m): Strictly Straight (Seedha)
                             • Zone 2 (s > 4.2m): Dynamic Quintic Obstacle Evasion Bend
                             • Horizon: Shortened to s_max = 14.5m (Urban road corridor)
                                                    |
                                                    v
                            [ Interactive Cockpit Dashboard & Video Suite ]
                            • Split Workspace (Vision + Calibrated AR | Radar BEV Scope)
                            • Telemetry Cards (Speed, Closest Obstacle, Locks, TTC)
                            • Synchronized 1080p MP4 Video Suite in D:\PPD2\Final_Videos\
```

---

## 2. Directory Structure

```
Autonomous_Vehicle_Complete_Flow/
│
├── dashboard.html            # Standalone Cockpit Dashboard UI (Zero server required)
├── dashboard_data.js         # Real-time telemetry, 77 GHz radar targets, and decisions
├── start_dashboard.bat       # 1-Click launcher to open dashboard in default browser
│
├── dashboard_frames/         # 150 synchronized, calibrated frames (d0, d1, d2)
│   ├── frame_0000.jpg
│   └── ...
│
├── models/                   # Pre-trained deep learning weights
│   ├── yolov8n.pt            # YOLOv8 PyTorch weights
│   ├── yolov8n.onnx          # YOLOv8 ONNX optimized model
│   ├── model2_lidar.onnx     # PointPillars 3D LiDAR detector ONNX
│   └── model3_moe_trajectory.onnx # Mixture-of-Experts (MoE) Trajectory Predictor ONNX
│
├── build_dashboard_assets.py                    # Pipeline asset builder & corridor extractor
├── compile_model1_video.py                      # Model 1 Camera video builder
├── generate_model2_bev_video.py                 # Model 2 1-to-1 LiDAR BEV video generator
├── generate_mmwave_radar_video.py               # 77 GHz FMCW mmWave Radar video generator
├── generate_multi_location_simulation_video.py  # Split-screen multi-modal simulation video generator
├── pipeline_perception_prediction_planning.py   # Full 3-model end-to-end Python pipeline
├── validate_model1_model2.py                    # Multi-sensor validation script
└── README_FLOW.md                               # Complete Flow Documentation (this file)
```

---

## 3. Sensor Modality Deep-Dive

### A. Model 1: Front Camera Vision (YOLOv8)
* **Hardware Representation**: 1080p Front Monocular Driving Camera ($1920 \times 1080$ RGB).
* **Task**: Real-time object detection across 7 Indian driving categories (`car`, `motorcycle`, `bus`, `truck`, `person`, `bicycle`, `auto_rickshaw`).
* **Inference Throughput**: $62.6\text{ FPS}$ ($16.0\text{ ms/frame}$) on NVIDIA GeForce RTX 2050 GPU.
* **Output Video**: High-definition pure camera vision without LiDAR or path overlay clutter (`model1_camera_vision_simulation.mp4`).

### B. Model 2: 3D LiDAR BEV Perception
* **Hardware Representation**: 32-Beam 3D LiDAR (Velodyne VLP-32C).
* **1-to-1 Spatial Sensor Fusion**: To prevent ghost boxes or unconstrained anchor bias, camera 2D detections are mapped to LiDAR point cloud clusters via pinhole perspective projection:
  $$ u = \frac{-Y \cdot f_x}{X} + c_x, \qquad v = \frac{-Z \cdot f_y}{X} + c_y $$
  where $f_x = f_y = 1250.0\text{ px}$, $c_x = 960.0\text{ px}$, $c_y = 540.0\text{ px}$.
* **Centroid Extraction**: Physical vehicle coordinates are extracted as the median of bounded points:
  $$ X_{\text{veh}} = \text{median}(X_{\text{points}}), \qquad Y_{\text{veh}} = \text{median}(Y_{\text{points}}) $$
* **Visual Style**: Clean dark aerospace BEV canvas, blue/indigo concentric laser scan rings, cyan obstacle reflection points, translucent coral/red 3D vehicle bounding boxes, bottom-center Ego vehicle with forward orientation arrow, and **ZERO text labels**.

### C. 77 GHz FMCW Automotive mmWave Radar Scope
* **Hardware Representation**: Texas Instruments AWR1843 (77–81 GHz FMCW, 3 Tx, 4 Rx MIMO Array).
* **Radar Range Equation**:
  $$ P_r = \frac{P_t \cdot G^2 \cdot \lambda^2 \cdot \sigma}{(4\pi)^3 \cdot R^4 \cdot L_{\text{atm}}} $$
  * Transmit Power: $P_t = 12.0\text{ dBm} \approx 15.8\text{ mW}$
  * Antenna Gain: $G = 24.0\text{ dBi}$
  * Center Frequency: $f_c = 77.0\text{ GHz} \implies \lambda \approx 3.9\text{ mm}$
  * Radar Cross Section (RCS, $\sigma$): Truck/Bus: $+25\text{ dBsm}$, Car: $+15\text{ dBsm}$, Two-Wheeler: $+8\text{ dBsm}$, Pedestrian: $+1\text{ dBsm}$.
* **Doppler Radial Velocity ($v_r$)**:
  $$ v_r = \frac{X \cdot v_x + Y \cdot v_y}{\sqrt{X^2 + Y^2}} = \frac{\lambda \cdot f_D}{2} $$
  * **Red Diamonds & Vectors**: Approaching / Closing targets ($v_r < -0.5\text{ m/s}$), representing higher collision risk.
  * **Green Diamonds & Vectors**: Receding / Opening targets ($v_r > +0.5\text{ m/s}$).
  * **Amber Speckle**: Stationary clutter and road guardrail returns ($v_r \approx 0\text{ m/s}$, ~60–80 targets/scan).
* **Technical Parameters Distinction**:
  $$ \text{Theoretical Range Resolution } (\Delta R = 3.75\text{ cm}) \neq \text{Measurement Noise } (\sigma_R = \pm 0.15\text{ m}) \neq \text{Doppler Resolution } (\Delta v = 0.21\text{ m/s}) $$

---

## 4. Model 3: Mixture-of-Experts (MoE) Trajectory Predictor

Model 3 is an advanced **Mixture-of-Experts GRU Neural Network** (`model3_moe_trajectory.onnx`) designed specifically for unstructured and heterogeneous Indian traffic.

### Architecture Overview

```
Past Trajectory [B, 8, 2] ──► Kinematic MLP (4 -> 32 -> 64) ──┐
                                                               │
Class ID [B, 1]           ──► Class Embedding (16-dim)       ──┼──► [Expert 1: Straight Cruising]  ──┐
                                                               │    [Expert 2: Braking / Stop]      ──┼──► Gated Trajectory Fusion
                                                               ├──► [Expert 3: Turning / Cut-in]    ──┤     Y_future = Σ (gi · Yi)
                                                               │    [Expert 4: Two-Wheeler Swerve] ──┘
Last Velocity + Summary   ─────────────────────────────────────┴──► Softmax Gating Network ───────────► Gating Weights [g1, g2, g3, g4]
```

### The 4 Specialized Experts

| Expert Module | Kinematic Specialization | Active Driving Scenario |
| :--- | :--- | :--- |
| **Expert 1** | **Straight Cruising (Constant Velocity)** | High-speed steady driving along straight highway and arterial corridors ($v_x \approx \text{const}, v_y \approx 0$). |
| **Expert 2** | **Deceleration / Braking / Stopping Mode** | Rapid slowdown at traffic intersections, pedestrian crossings, or sudden braking by a lead vehicle. |
| **Expert 3** | **Lateral Lane Change & Turning** | Vehicle turning at junctions, overtaking manoeuvres, and aggressive lane cut-ins ($v_y \neq 0$). |
| **Expert 4** | **Agile Swerving (Two-Wheelers & Pedestrians)** | High-frequency lateral shifts and agile trajectory deviations characteristic of Indian motorcycles, auto-rickshaws, and pedestrians. |

### Softmax Gating & Trajectory Synthesis
The Gating Network evaluates the motion summary, class embedding, and latest velocity to dynamically distribute confidence weights:

$$ g_i = \frac{\exp(z_i)}{\sum_{j=1}^{4} \exp(z_j)}, \qquad \sum_{i=1}^{4} g_i = 1 $$

The 12-step future prediction ($2.4\text{ seconds}$ horizon at $\Delta t = 0.2\text{s}$) is computed as:

$$ \hat{Y}_{\text{future}} = \sum_{i=1}^{4} g_i \cdot \hat{Y}_{\text{Expert } i} $$

---

## 5. Path Planning: 3-Zone Frenet Optimal Evasion

To ensure natural, comfortable steering without sudden jerky twitches, the Frenet lateral trajectory $d(s)$ is formulated as a quintic polynomial:

$$ d(s) = d_0 + \Delta d \cdot \left( 10\tau^3 - 15\tau^4 + 6\tau^5 \right) $$

* **Zone 1 ($s \le 4.2\text{ m}$):** $d(s) = 0.0\text{ m}$ (**Strictly Straight in front of ego vehicle**). Prevents the vehicle from steering prematurely while the obstacle is still down the road.
* **Zone 2 ($4.2 < s \le s_{\text{obs}} + 2.5\text{ m}$):** Evasion progression $\tau = \frac{s - 4.2}{s_{\text{obs}} - 2.0} \in [0, 1]$.
* **Lookahead Horizon:** Shortened to $s_{\max} = 14.5\text{ m}$ for tight urban corridor compliance.

---

## 6. Generated Video Portfolio (`Final_Videos/`)

All processed video streams are centralized in **`D:\PPD2\Final_Videos\`**:

| Video File | Sensor Modality | Resolution | FPS | Duration | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **model1_camera_vision_simulation.mp4** | Monocular Camera | 1920×1080 | 10.0 | 15.0s | Pure YOLOv8 2D vision detections (Green boxes). |
| **model2_lidar_bev_simulation.mp4** | 3D LiDAR BEV | 1920×1080 | 10.0 | 15.0s | 1-to-1 Spatial Sync, concentric laser scan rings, translucent coral boxes, zero text. |
| **mmwave_radar_bev_simulation.mp4** | 77 GHz mmWave Radar | 1920×1080 | 10.0 | 15.0s | TI AWR1843 Doppler BEV Scope, velocity vectors, target telemetry HUD. |
| **model1_all_6168_complete.mp4** | Monocular Camera | 1920×1080 | 10.0 | 10.28 min | Full dataset run across all 6,168 frames of Drives 0, 1, and 2. |

---

## 7. Execution Guide

### 1. Launch Cockpit Dashboard
```cmd
start_dashboard.bat
```
*Opens `dashboard.html` in your default web browser.*

### 2. Generate Model 2 LiDAR BEV Video
```bash
python generate_model2_bev_video.py
```

### 3. Generate 77 GHz mmWave Radar Scope Video
```bash
python generate_mmwave_radar_video.py
```

### 4. Run End-to-End Perception, Prediction & Planning Pipeline
```bash
python pipeline_perception_prediction_planning.py
```
