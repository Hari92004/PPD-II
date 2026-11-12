# Autonomous Vehicle Multi-Modal Perception, Prediction, Planning & Cockpit Flow

**Complete Standalone Flow Package for Indian Road Driving (IDD Synchronized Dataset)**

---

## 1. System Flow Overview

```
                                [ Synchronized Indian Driving Dataset (IDD) ]
                                          /                       \
                                         /                         \
                        [ Monocular Camera (Front) ]        [ LiDAR 3D Point Cloud ]
                                     |                                 |
                                     v                                 v
                        [ Model 1: YOLOv8 2D ]           [ Model 2: PointPillars 3D ]
                        (Bounding Boxes: Green)          (Corridor Search: Blue <25m)
                                     \                                 /
                                      \                               /
                                       v                             v
                                  [ Spatial Camera-LiDAR Calibration Fusion ]
                                                     |
                                                     v
                                  [ 77 GHz mmWave FMCW Radar Simulation ]
                                  (Doppler vr, Range Rings, RCS, BEV Scope)
                                                     |
                                                     v
                             [ 3-Zone Frenet Quintic Path Planner & Evasion ]
                             • Zone 1 (s <= 4.2m): Strictly Straight (Seedha)
                             • Zone 2 (s > 4.2m): Dynamic Quintic Obstacle Evasion Bend
                             • Horizon: Shortened to s_max = 14.5m (Urban corridor)
                                                     |
                                                     v
                            [ Interactive Cockpit Dashboard (dashboard.html) ]
                            • Split Workspace (Vision + Calibrated AR | Radar BEV Scope)
                            • Telemetry Cards (Speed, Closest Obstacle, Locks, TTC)
                            • Live Scrubber, Speed Controls & Drive Switcher (d0, d1, d2)
```

---

## 2. Directory Structure

```
Autonomous_Vehicle_Complete_Flow/
│
├── dashboard.html            # Standalone Cockpit Dashboard UI (Zero server needed)
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
├── generate_multi_location_simulation_video.py  # Split-screen simulation video generator
├── pipeline_perception_prediction_planning.py   # Full 3-model end-to-end Python pipeline
├── validate_model1_model2.py                    # Multi-sensor validation script
└── README_FLOW.md                               # Flow documentation (this file)
```

---

## 3. Mathematical Formulations

### A. 77 GHz mmWave FMCW Radar Range Equation
The received radar power $P_r$ for an obstacle at range $R$ with Radar Cross Section (RCS) $\sigma$ is governed by:

$$ P_r = \frac{P_t G^2 \lambda^2 \sigma}{(4\pi)^3 R^4} $$

* $P_t$ = Transmit power ($15\text{ dBm} \approx 31.6\text{ mW}$)
* $G$ = Antenna gain ($18\text{ dBi}$)
* $\lambda = \frac{c}{f_c} \approx 3.9\text{ mm}$ at $f_c = 77\text{ GHz}$
* $\sigma$ = RCS: Car $\approx 14.5\text{ dBsm}$, Two-Wheeler $\approx 4.5\text{ dBsm}$, Truck/Bus $\approx 22.0\text{ dBsm}$
* $R$ = Radial range to obstacle (meters)

### B. Doppler Radial Velocity
$$ v_r = \frac{\vec{x}_{\text{rel}} \cdot \vec{R}}{R} = \cos(\theta) \cdot (v_{\text{target}} - v_{\text{ego}}) $$

* Negative $v_r$ indicates an approaching target (Collision Risk / Lower TTC).
* Positive $v_r$ indicates a receding target.

### C. 3-Zone Frenet Quintic Dynamic Evasion
To ensure smooth steering curvature and zero initial lateral jerk:

$$ d(s) = d_0 + \Delta d \cdot \left( 10\tau^3 - 15\tau^4 + 6\tau^5 \right) $$

* **Zone 1 ($s \le 4.2\text{ m}$):** $d(s) = 0.0\text{ m}$ (**Strictly Straight in front of ego vehicle**).
* **Zone 2 ($4.2 < s \le s_{\text{obs}} + 2.5\text{ m}$):** Evasion progression $\tau = \frac{s - 4.2}{s_{\text{obs}} - 2.0} \in [0, 1]$.
* **Lookahead Horizon:** $s_{\max} = 14.5\text{ m}$ (Shortened for urban road corridor focus).

---

## 4. How to Run

1. **Launch Interactive Dashboard:**
   * Double-click `start_dashboard.bat` OR open `dashboard.html` in Chrome/Edge/Firefox.
   * Play/pause with `Space` or button, scrub frames, switch between Drive 0 (Urban), Drive 1 (Arterial), and Drive 2 (Highway).

2. **Re-generate Dashboard Assets:**
   ```bash
   python build_dashboard_assets.py
   ```

3. **Run Full Video Simulation Generator:**
   ```bash
   python generate_multi_location_simulation_video.py
   ```

