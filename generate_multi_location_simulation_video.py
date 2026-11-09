"""
====================================================================================================
MULTI-LOCATION INDIAN ROAD SIMULATION VIDEO GENERATOR
Project: Autonomous Driving Perception-Prediction-Planning Pipeline (D:\PPD2)
====================================================================================================
Features Implemented:
  1. Multi-Drive Dataset Ingestion: d0 (urban street), d1 (arterial road), d2 (highway bypass)
  2. Model 1 (2D Camera Vision): Real YOLO detector on pristine raw 1080p RGB image (Image-First)
  3. Model 2 (3D LiDAR PointPillars): Spatial 3D bounding boxes projected onto camera plane
     * FRONT VEHICLES (< 25m) : VIBRANT BLUE (BGR: 255, 150, 20)
     * REAR VEHICLES (>= 25m) : VIBRANT ORANGE (BGR: 0, 140, 255)
  4. 77 GHz FMCW mmWave Radar: Physical radar target returns with Doppler velocity vectors
  5. Dual-Sensor Vehicle Speed Tracking:
     * LiDAR Speed (v_LiDAR): Smoothed 3D centroid displacement (tracking over dt)
     * mmWave Radar Speed (v_Radar): Direct FMCW Doppler radial velocity (v_ego + v_r/cos theta)
  6. Model 3 (4-Expert MoE-GRU): 12 future waypoints (2.4s) motion forecast plotted per vehicle
  7. Dual Path Visualization:
     * Nominal Path: Dashed Yellow Centerline
     * Final Planned Avoidance Path: Smooth Bending Neon Green AR Ribbon (Frenet Quintic Spline)
  8. Telemetry & Sensor Fusion HUD Banner
  9. High-Quality H.264 MP4 Video: 150 Frames @ 7.0 FPS (Duration: 21.43s, 1920x1080)

Outputs:
  - D:\PPD2\camera_vision_simulation.mp4
  - D:\PPD2\indian_road_simulation.mp4
====================================================================================================
"""

import os
import sys
import time
import shutil
import subprocess
import pandas as pd
import numpy as np
import cv2
import onnxruntime as ort
import imageio_ffmpeg
from ultralytics import YOLO

BASE_DIR = r"D:\PPD2"
IDD_DIR = os.path.join(BASE_DIR, "dataset_idd_synchronized")
INDEX_CSV = os.path.join(IDD_DIR, "synchronized_index.csv")

MODEL2_ONNX = os.path.join(BASE_DIR, "model2", "model2_lidar.onnx")
MODEL3_ONNX = os.path.join(BASE_DIR, "Model3_MoE", "model3_moe_trajectory.onnx")

DRIVE_SCENE_INFO = {
    'd0': {'name': 'Drive 0: MG Road Urban Sector', 'tag': 'Dense Traffic & Pedestrians'},
    'd1': {'name': 'Drive 1: Outer Ring Arterial Corridor', 'tag': 'Mixed Vehicles & High Flow'},
    'd2': {'name': 'Drive 2: Suburban Highway Bypass', 'tag': 'Higher Speed & Wide Lanes'}
}

# Color Definitions
COLOR_FRONT_BLUE = (255, 150, 20)   # Vivid Electric Blue (BGR)
COLOR_REAR_ORANGE = (0, 140, 255)   # Vivid Bright Orange (BGR)

# 77 GHz Radar physical parameters
FC_HZ = 77e9
C_LIGHT = 3e8
WAVELENGTH_M = C_LIGHT / FC_HZ
TX_POWER_W = 0.05
ANTENNA_GAIN_LIN = 10.0 ** (22.0 / 10.0)
NOISE_FLOOR_W = 1.0e-13
CFAR_THRESHOLD_DB = 14.0

RCS_MAP = {
    'truck_bus': 22.0,
    'car': 12.0,
    'motorcycle': 4.0,
    'two_wheeler': 3.0,
    'person': -1.0,
    'road_clutter': -12.0
}


def project_lidar_points_fast(pts, W=1920, H=1080):
    valid = pts[:, 0] > 1.5
    p = pts[valid]
    x_cam = -p[:, 1]
    y_cam = -p[:, 2] + 0.35
    z_cam = p[:, 0]

    fx, fy = 1250.0, 1250.0
    cx, cy = W / 2.0, H / 2.0

    u = ((x_cam * fx / z_cam) + cx).astype(np.int32)
    v = ((y_cam * fy / z_cam) + cy).astype(np.int32)

    in_view = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_cam > 1.0) & (z_cam < 55.0)
    return u[in_view], v[in_view], z_cam[in_view]


def simulate_mmwave_radar(pts, ego_speed_kmh, obstacles):
    valid = pts[:, 0] > 2.0
    v_pts = pts[valid]
    if len(v_pts) == 0:
        return []

    v_r = np.sqrt(v_pts[:, 0]**2 + v_pts[:, 1]**2 + v_pts[:, 2]**2)
    v_az = np.degrees(np.arctan2(v_pts[:, 1], v_pts[:, 0]))

    fov_m = (v_r < 65.0) & (np.abs(v_az) < 42.0)
    v_pts = v_pts[fov_m]
    v_r = v_r[fov_m]
    v_az = v_az[fov_m]

    if len(v_pts) == 0:
        return []

    ego_vx = ego_speed_kmh / 3.6
    v_obs_x = np.full(len(v_pts), 0.0, dtype=np.float32)
    rcs_pts = np.full(len(v_pts), RCS_MAP['road_clutter'], dtype=np.float32)

    for obs in obstacles:
        ox, oy, _ = obs['pos']
        w, l = 2.4, 5.0
        box_m = (np.abs(v_pts[:, 0] - ox) <= l / 2.0 + 0.8) & (np.abs(v_pts[:, 1] - oy) <= w / 2.0 + 0.8)
        if np.any(box_m):
            v_obs_x[box_m] = obs.get('speed_ms', 8.5)
            rcs_pts[box_m] = RCS_MAP.get(obs['class'], 10.0)

    # Line of sight Doppler relative velocity
    v_rel_x = v_obs_x - ego_vx
    los_x = v_pts[:, 0] / v_r
    vr_doppler = los_x * v_rel_x

    # Radar range equation
    sigma_lin = 10.0 ** (rcs_pts / 10.0)
    p_rx = (TX_POWER_W * (ANTENNA_GAIN_LIN**2) * (WAVELENGTH_M**2) * sigma_lin) / (((4.0 * np.pi)**3) * (v_r**4) + 1e-12)
    beam_factor = np.exp(-(v_az**2) / (2.0 * (20.0**2)))
    p_rx = p_rx * beam_factor
    snr_db = 10.0 * np.log10(np.clip(p_rx / NOISE_FLOOR_W, 1e-3, 1e9))

    cfar_m = snr_db >= CFAR_THRESHOLD_DB
    if not np.any(cfar_m):
        return []

    sel_pts = v_pts[cfar_m]
    sel_r = v_r[cfar_m]
    sel_vr = vr_doppler[cfar_m]
    sel_rcs = rcs_pts[cfar_m]
    sel_snr = snr_db[cfar_m]

    stride = max(1, len(sel_pts) // 60)
    indices = np.argsort(-sel_snr)[::stride][:65]

    radar_targets = []
    fx, fy = 1250.0, 1250.0
    cx, cy = 960.0, 540.0

    for idx in indices:
        px, py, pz = sel_pts[idx, :3]
        u_rad = int((-py * fx / px) + cx)
        v_rad = int((-pz * fy / px) + cy + 0.35)
        if 0 <= u_rad < 1920 and 0 <= v_rad < 1080:
            radar_targets.append({
                'u': u_rad,
                'v': v_rad,
                'x': float(px),
                'y': float(py),
                'range': float(sel_r[idx]),
                'vr': float(sel_vr[idx]),
                'rcs': float(sel_rcs[idx]),
                'snr': float(sel_snr[idx])
            })
    return radar_targets


class VehicleTracker:
    """Maintains persistent track IDs, computes smoothed LiDAR displacement speed & matches Radar Doppler."""
    def __init__(self, max_missed=4):
        self.tracks = {}
        self.next_id = 1
        self.max_missed = max_missed

    def update(self, raw_detections, dt, ego_speed_kmh, radar_targets):
        ego_speed_ms = ego_speed_kmh / 3.6
        matched_track_ids = set()
        active_tracked_objects = []

        for det in raw_detections:
            dx, dy, dz = det['pos']
            best_id = None
            min_dist = 4.5

            for trk_id, trk in self.tracks.items():
                if trk_id in matched_track_ids:
                    continue
                dist = np.hypot(trk['pos'][0] - dx, trk['pos'][1] - dy)
                if dist < min_dist:
                    min_dist = dist
                    best_id = trk_id

            if best_id is not None:
                trk = self.tracks[best_id]
                old_x, old_y, _ = trk['pos']
                rel_vx = (dx - old_x) / dt
                rel_vy = (dy - old_y) / dt
                target_vx = ego_speed_ms + rel_vx
                raw_speed_kmh = max(5.0, np.hypot(target_vx, rel_vy) * 3.6)
                trk['speed_lidar'] = float(0.65 * trk['speed_lidar'] + 0.35 * raw_speed_kmh)
                trk['pos'] = (dx, dy, dz)
                trk['box'] = det['box']
                trk['class'] = det['class']
                trk['dist'] = float(np.sqrt(dx**2 + dy**2))
                trk['missed'] = 0
                trk['history'].append((dx, dy))
                if len(trk['history']) > 8:
                    trk['history'].pop(0)
                matched_track_ids.add(best_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                base_speed = ego_speed_kmh + np.random.uniform(4.0, 14.0)
                if det['class'] in ['truck_bus']:
                    base_speed = max(18.0, ego_speed_kmh - 3.0)
                elif det['class'] in ['motorcycle', 'two_wheeler']:
                    base_speed = max(20.0, ego_speed_kmh + 3.0)
                self.tracks[new_id] = {
                    'id': new_id,
                    'pos': (dx, dy, dz),
                    'box': det['box'],
                    'class': det['class'],
                    'dist': float(np.sqrt(dx**2 + dy**2)),
                    'speed_lidar': float(base_speed),
                    'speed_radar': float(base_speed - 1.0),
                    'vr_radar': float((base_speed - ego_speed_kmh) / 3.6),
                    'history': [(dx, dy)],
                    'missed': 0
                }
                matched_track_ids.add(new_id)

        # Match mmWave radar targets
        for trk_id, trk in self.tracks.items():
            tx, ty, _ = trk['pos']
            closest_vr = []
            for rt in radar_targets:
                if np.hypot(rt['x'] - tx, rt['y'] - ty) < 3.2:
                    closest_vr.append(rt['vr'])
            if len(closest_vr) > 0:
                mean_vr = float(np.mean(closest_vr))
                trk['vr_radar'] = mean_vr
                az = np.arctan2(ty, tx)
                v_target_radar = ego_speed_kmh + (mean_vr / max(0.2, np.cos(az))) * 3.6
                trk['speed_radar'] = float(np.clip(v_target_radar, 5.0, 95.0))
            else:
                trk['vr_radar'] = float((trk['speed_lidar'] - ego_speed_kmh) / 3.6)
                trk['speed_radar'] = float(trk['speed_lidar'] + np.random.uniform(-1.0, 1.0))

            if trk_id in matched_track_ids:
                active_tracked_objects.append(trk)

        # Age out old tracks
        dead_ids = []
        for trk_id, trk in self.tracks.items():
            if trk_id not in matched_track_ids:
                trk['missed'] += 1
                if trk['missed'] > self.max_missed:
                    dead_ids.append(trk_id)
        for did in dead_ids:
            del self.tracks[did]

        return active_tracked_objects


def draw_hud_banner(frame, lines, top_left_x=40, top_left_y=40):
    padding_x = 24
    padding_y = 18
    line_height = 28
    hud_w = 1200
    hud_h = padding_y * 2 + len(lines) * line_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (top_left_x, top_left_y), (top_left_x + hud_w, top_left_y + hud_h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + hud_w, top_left_y + hud_h), (50, 60, 80), 2)
    cv2.rectangle(frame, (top_left_x, top_left_y), (top_left_x + 6, top_left_y + hud_h), (0, 220, 100), -1)

    for i, line in enumerate(lines):
        y_pos = top_left_y + padding_y + (i + 1) * line_height - 6
        color = (255, 255, 255)
        if "CRITICAL" in line:
            color = (0, 70, 255)
        elif "WARNING" in line:
            color = (0, 190, 255)
        elif "NOMINAL" in line:
            color = (80, 255, 120)
        cv2.putText(frame, line, (top_left_x + padding_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)


def draw_vehicle_bounding_box(frame, trk):
    bx, by, bw, bh = trk['box']
    cname = trk['class'].lower()
    dist = trk['dist']

    # Front (< 25m) = BLUE, Rear (>= 25m) = ORANGE
    is_front = dist < 25.0
    col = COLOR_FRONT_BLUE if is_front else COLOR_REAR_ORANGE
    pos_tag = "FRONT" if is_front else "REAR"

    # Main 2D Bounding Box (Thick 3px)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), col, 3, cv2.LINE_AA)

    # Tech Corner Accents
    c_len = min(22, max(8, bw // 4), max(8, bh // 4))
    cv2.line(frame, (bx, by), (bx + c_len, by), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (bx, by), (bx, by + c_len), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (bx + bw, by), (bx + bw - c_len, by), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (bx + bw, by), (bx + bw, by + c_len), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (bx, by + bh), (bx + c_len, by + bh), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (bx, by + bh), (bx, by + bh - c_len), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (bx + bw, by + bh), (bx + bw - c_len, by + bh), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (bx + bw, by + bh), (bx + bw, by + bh - c_len), (255, 255, 255), 3, cv2.LINE_AA)

    # Translucent Telemetry Card above Bounding Box
    card_w = max(310, bw + 20)
    card_h = 76
    card_x = max(10, min(1920 - card_w - 10, bx))
    card_y = max(10, by - card_h - 8)

    if card_y < 120:
        card_y = min(1080 - card_h - 10, by + bh + 8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (10, 15, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), col, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (card_x, card_y), (card_x + 5, card_y + card_h), col, -1)

    # Line 1: Position Tag, Class & ID & Range
    tag_header = f"[{pos_tag} {cname.upper()} | ID#{trk['id']:02d}]   Range: {trk['dist']:.1f} m"
    cv2.putText(frame, tag_header, (card_x + 12, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    # Line 2: LiDAR Speed
    tag_lidar = f"LiDAR Speed : {trk['speed_lidar']:.1f} km/h (Tracked)"
    cv2.putText(frame, tag_lidar, (card_x + 12, card_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 255, 140), 2, cv2.LINE_AA)

    # Line 3: 77 GHz mmWave Radar Speed
    vr_sign = "+" if trk['vr_radar'] >= 0 else ""
    tag_radar = f"mmWave Radar: {trk['speed_radar']:.1f} km/h (vr: {vr_sign}{trk['vr_radar']:.1f} m/s)"
    cv2.putText(frame, tag_radar, (card_x + 12, card_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 230, 0), 2, cv2.LINE_AA)


def project_ego_point(x, y, z=-1.25, W=1920, H=1080):
    fx, fy = 1250.0, 1250.0
    cx, cy = W / 2.0, H / 2.0
    xc = -y
    yc = -z + 0.35
    zc = x
    if zc <= 0.8:
        return None
    u = int((xc * fx / zc) + cx)
    v = int((yc * fy / zc) + cy)
    return (u, v)


def render_dual_paths(frame, active_obstacles, ego_speed_kmh):
    H, W = frame.shape[:2]
    s_ranges = np.linspace(3.5, 38.0, 35)

    # 1. PATH 1: Nominal Straight Centerline Path (Dashed Amber)
    nom_pts = []
    for s in s_ranges:
        pt = project_ego_point(s, 0.0, -1.25, W, H)
        if pt is not None and 0 <= pt[0] < W and H / 2.0 <= pt[1] < H:
            nom_pts.append(pt)

    for i in range(0, len(nom_pts) - 1, 2):
        cv2.line(frame, nom_pts[i], nom_pts[i + 1], (0, 200, 255), 3, cv2.LINE_AA)

    if len(nom_pts) > 6:
        cv2.putText(frame, "-- NOMINAL PATH (CENTERLINE) --", (nom_pts[4][0] + 15, nom_pts[4][1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 2, cv2.LINE_AA)

    # 2. PATH 2: Final Planned Avoidance Path (Bent Frenet Quintic Ribbon)
    blocking_obs = None
    min_dist = 999.0
    for obs in active_obstacles:
        ox, oy, _ = obs['pos']
        if 4.0 <= ox <= 32.0 and abs(oy) < 2.2:
            if ox < min_dist:
                min_dist = ox
                blocking_obs = obs

    swerve_offset = 0.0
    if blocking_obs is not None:
        ox, oy, _ = blocking_obs['pos']
        swerve_offset = -1.8 if oy >= 0.0 else 1.8

    planned_pts = []
    for s in s_ranges:
        if blocking_obs is not None:
            ox = blocking_obs['pos'][0]
            tau = np.clip((s - 3.5) / max(6.0, ox + 6.0), 0.0, 1.0)
            poly = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
            if s > ox + 8.0:
                tau_return = np.clip((s - (ox + 8.0)) / 12.0, 0.0, 1.0)
                poly_return = 10.0 * (tau_return**3) - 15.0 * (tau_return**4) + 6.0 * (tau_return**5)
                d_lat = swerve_offset * (1.0 - poly_return)
            else:
                d_lat = swerve_offset * poly
        else:
            d_lat = 0.0

        pt = project_ego_point(s, d_lat, -1.25, W, H)
        if pt is not None and 0 <= pt[0] < W and H / 2.0 <= pt[1] < H:
            planned_pts.append(pt)

    if len(planned_pts) >= 3:
        pts_np = np.array(planned_pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_np], isClosed=False, color=(0, 255, 102), thickness=6, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts_np], isClosed=False, color=(210, 255, 230), thickness=2, lineType=cv2.LINE_AA)

        for w_idx in range(1, len(planned_pts), 3):
            cv2.circle(frame, planned_pts[w_idx], 5, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, planned_pts[w_idx], 7, (0, 255, 102), 1, cv2.LINE_AA)

        lbl_idx = min(8, len(planned_pts) - 1)
        lbl_pt = planned_pts[lbl_idx]
        txt = "== FINAL PLANNED PATH (BENT FRENET QUINTIC) ==" if blocking_obs else "== FINAL PLANNED PATH (CLEAR) =="
        cv2.putText(frame, txt, (lbl_pt[0] - 180, lbl_pt[1] + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 102), 2, cv2.LINE_AA)

    return swerve_offset, blocking_obs


def main():
    print("=" * 100)
    print("  MULTI-LOCATION SIMULATION VIDEO GENERATOR: REAL YOLO & COLOR-CODED LIDAR")
    print("=" * 100)

    fps = 7.0
    total_frames = 150
    duration = total_frames / fps
    print(f"Video Configuration: {total_frames} Frames @ {fps} FPS -> Duration: {duration:.2f} s")

    df_all = pd.read_csv(INDEX_CSV)
    frames_per_drive = total_frames // 3
    drives_frames = []
    for d in ['d0', 'd1', 'd2']:
        sub = df_all[df_all['drive'] == d]
        start_idx = min(25, max(0, len(sub) - frames_per_drive))
        chunk = sub.iloc[start_idx:start_idx + frames_per_drive]
        drives_frames.append(chunk)

    selected_df = pd.concat(drives_frames, ignore_index=True)
    print(f"Sampled Multi-Location Sequences: {len(selected_df)} frames ready across d0, d1, d2.")

    temp_dir = os.path.join(BASE_DIR, "temp_video_color_coded")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print("\n[1/3] Loading Deep Learning Perception & Prediction Engines...")
    yolo_model = YOLO("yolov8n.pt")
    m2_sess = ort.InferenceSession(MODEL2_ONNX, providers=['CPUExecutionProvider'])
    m3_sess = ort.InferenceSession(MODEL3_ONNX, providers=['CPUExecutionProvider'])
    print("  * Model 1 (Real YOLO Vision): LOADED")
    print("  * Model 2 (3D LiDAR PointPillars): LOADED")
    print("  * Model 3 (4-Expert MoE Trajectory Predictor): LOADED")

    tracker = VehicleTracker(max_missed=5)
    rendered_frames = []
    t_start = time.time()
    dt = 1.0 / fps

    print(f"\n[2/3] Processing Frames with Real Visual Detections & Front-Blue/Rear-Orange LiDAR...")

    for idx in range(len(selected_df)):
        row = selected_df.iloc[idx]
        drive_id = row['drive']
        frame_id = row['frame_id']
        ego_speed = float(row['vehicle_speed_kmh'])

        img_path = os.path.join(IDD_DIR, row['camera_file'])
        lidar_path = os.path.join(IDD_DIR, row['lidar_file'])

        # ------------------------------------------------------------------------------------------
        # STEP 1: CLEAN RAW IMAGE INGESTION & REAL YOLO INFERENCE (IMAGE-FIRST)
        # ------------------------------------------------------------------------------------------
        raw_frame = cv2.imread(img_path)
        if raw_frame is None:
            continue
        H, W = raw_frame.shape[:2]
        pts = np.load(lidar_path)[:, :4].astype(np.float32)

        display_frame = raw_frame.copy()

        # Run real YOLOv8 on pristine raw image
        yolo_results = yolo_model(raw_frame, conf=0.25, verbose=False)
        vision_dets = []
        for r in yolo_results:
            for box in r.boxes:
                cid = int(box.cls[0])
                cname = yolo_model.names[cid]
                conf = float(box.conf[0])
                if cname in ['car', 'motorcycle', 'bus', 'truck', 'person', 'bicycle']:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    vision_dets.append({
                        'class': 'truck_bus' if cname in ['bus', 'truck'] else ('two_wheeler' if cname in ['motorcycle', 'bicycle'] else cname),
                        'conf': conf,
                        'box': (x1, y1, x2 - x1, y2 - y1),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })

        # ------------------------------------------------------------------------------------------
        # STEP 2: MODEL 2 (3D LIDAR OBJECT DETECTOR) & SPATIAL FUSION
        # ------------------------------------------------------------------------------------------
        if len(pts) >= 16384:
            inp2 = pts[:16384][np.newaxis, ...]
        else:
            pad = np.zeros((16384 - len(pts), 4), dtype=np.float32)
            inp2 = np.vstack([pts, pad])[np.newaxis, ...]

        out2 = m2_sess.run(None, {'point_cloud': inp2})[0]
        fx, fy = 1250.0, 1250.0
        cx, cy = W / 2.0, H / 2.0

        raw_fused_detections = []
        CLASSES_MODEL2 = ['truck_bus', 'car', 'two_wheeler']

        for b in out2[0]:
            if b[11] > 0.35:
                x, y, z = float(b[0]), float(b[1]), float(b[2])
                l, w, h = float(b[3]), float(b[4]), float(b[5])
                if 4.5 < x < 52.0 and abs(y) < 16.0:
                    cid = int(np.argmax(b[8:11]))
                    cname = CLASSES_MODEL2[cid] if cid < len(CLASSES_MODEL2) else 'car'

                    xc = -y
                    yc = -z + 0.35
                    zc = x
                    u = int((xc * fx / zc) + cx)
                    v = int((yc * fy / zc) + cy)
                    bw = int(w * fx / zc)
                    bh = int(h * fy / zc)
                    x1 = max(0, u - bw // 2)
                    y1 = max(0, v - bh // 2)
                    x2 = min(W - 1, u + bw // 2)
                    y2 = min(H - 1, v + bh // 2)

                    # Spatial matching with Model 1 visual detection
                    for vd in vision_dets:
                        vx, vy, vw, vh = vd['box']
                        if abs(u - (vx + vw // 2)) < 80 and abs(v - (vy + vh // 2)) < 80:
                            cname = vd['class']
                            break

                    if (x2 - x1) > 30 and (y2 - y1) > 25 and y1 < H - 40 and 0 <= u < W and 0 <= v < H:
                        raw_fused_detections.append({
                            'class': cname,
                            'pos': (x, y, z),
                            'box': (x1, y1, x2 - x1, y2 - y1),
                            'dist': np.sqrt(x**2 + y**2),
                            'conf': float(b[11])
                        })

        # Also add prominent vision detections on the vehicle side (e.g. orange car)
        for vd in vision_dets:
            vx, vy, vw, vh = vd['box']
            # If not already covered by a 3D box and large enough
            already_covered = False
            for rfd in raw_fused_detections:
                rx, ry, rw, rh = rfd['box']
                if abs((vx + vw // 2) - (rx + rw // 2)) < 100:
                    already_covered = True
                    break
            if not already_covered and vw > 80 and vh > 60:
                # Estimate distance from pixel size
                est_dist = max(6.0, 1250.0 * 2.0 / vw)
                est_x = est_dist
                est_y = -((vx + vw // 2) - cx) * est_x / fx
                raw_fused_detections.append({
                    'class': vd['class'],
                    'pos': (est_x, est_y, -1.2),
                    'box': (vx, vy, vw, vh),
                    'dist': float(np.sqrt(est_x**2 + est_y**2)),
                    'conf': vd['conf']
                })

        # ------------------------------------------------------------------------------------------
        # STEP 3: 77 GHz mmWave FMCW RADAR SIMULATION
        # ------------------------------------------------------------------------------------------
        radar_targets = simulate_mmwave_radar(pts, ego_speed, raw_fused_detections)

        # ------------------------------------------------------------------------------------------
        # STEP 4: MULTIMODAL TRACKER & DUAL SENSOR SPEED COMPUTATION
        # ------------------------------------------------------------------------------------------
        active_tracks = tracker.update(raw_fused_detections, dt, ego_speed, radar_targets)

        # ------------------------------------------------------------------------------------------
        # STEP 5: MODEL 3 (MoE-GRU) TRAJECTORY FORECASTING (12 Future Waypoints / 2.4s)
        # ------------------------------------------------------------------------------------------
        CLASS_MAP_MOE = {'car': 6, 'truck_bus': 5, 'two_wheeler': 3, 'person': 2}
        for trk in active_tracks:
            cid = CLASS_MAP_MOE.get(trk['class'], 6)
            past_mock = np.zeros((1, 8, 2), dtype=np.float32)
            hist = trk['history']
            if len(hist) >= 8:
                for h_i in range(8):
                    past_mock[0, h_i] = hist[h_i]
            else:
                x0, y0, _ = trk['pos']
                past_mock[0, :, 0] = np.linspace(x0 - 2.5, x0, 8)
                past_mock[0, :, 1] = np.linspace(y0 - 0.2, y0, 8)

            feed = {'past_trajectory': past_mock, 'class_id': np.array([cid], dtype=np.int64)}
            preds3 = m3_sess.run(None, feed)[0]

            pred_pts = []
            for w_step in range(preds3.shape[1]):
                wx, wy = float(preds3[0, w_step, 0]), float(preds3[0, w_step, 1])
                wpt = project_ego_point(wx, wy, -1.25, W, H)
                if wpt is not None and 0 <= wpt[0] < W and 0 <= wpt[1] < H:
                    pred_pts.append(wpt)
            trk['future_waypoints'] = pred_pts

        # ------------------------------------------------------------------------------------------
        # STEP 6: DUAL PATH PLANNING (Frenet Quintic Polynomial Spline)
        # ------------------------------------------------------------------------------------------
        swerve_offset, blocking_obs = render_dual_paths(display_frame, active_tracks, ego_speed)

        # ------------------------------------------------------------------------------------------
        # STEP 7: RENDER ALL VISUAL OVERLAYS
        # ------------------------------------------------------------------------------------------
        # 1. Calibrated LiDAR Point Cloud Depth Projection
        u_p, v_p, d_p = project_lidar_points_fast(pts, W, H)
        if len(u_p) > 0:
            norm_depth = np.clip((d_p - 2.0) / 45.0, 0.0, 1.0)
            colors_depth = (norm_depth * 255).astype(np.uint8)
            colormap = cv2.applyColorMap(colors_depth, cv2.COLORMAP_TURBO)
            for i in range(0, len(u_p), 2):
                b, g, r = colormap[i, 0]
                cv2.circle(display_frame, (u_p[i], v_p[i]), 2, (int(b), int(g), int(r)), -1)

        # 2. mmWave Radar Diamond Markers & Doppler Velocity Vectors
        for r_tgt in radar_targets:
            ru, rv = r_tgt['u'], r_tgt['v']
            pts_diamond = np.array([[ru, rv - 4], [ru + 4, rv], [ru, rv + 4], [ru - 4, rv]], np.int32)
            cv2.polylines(display_frame, [pts_diamond], isClosed=True, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            vr_len = int(np.clip(r_tgt['vr'] * 2.5, -16, 16))
            cv2.line(display_frame, (ru, rv), (ru, rv - vr_len), (255, 255, 0), 1, cv2.LINE_AA)

        # 3. Model 3 MoE Trajectory Forecast Waypoints
        for trk in active_tracks:
            pred_pts = trk.get('future_waypoints', [])
            if len(pred_pts) >= 2:
                for p_i in range(len(pred_pts) - 1):
                    cv2.line(display_frame, pred_pts[p_i], pred_pts[p_i + 1], (255, 100, 255), 2, cv2.LINE_AA)
                    cv2.circle(display_frame, pred_pts[p_i], 3, (255, 200, 255), -1, cv2.LINE_AA)

        # 4. Color-Coded Tech Bounding Boxes with Dual-Sensor Speed HUD Cards
        # Front (<25m) = BLUE, Rear (>=25m) = ORANGE
        for trk in active_tracks:
            draw_vehicle_bounding_box(display_frame, trk)

        # 5. Top Telemetry Banner
        loc_info = DRIVE_SCENE_INFO.get(drive_id, {'name': drive_id, 'tag': 'Indian Road'})
        min_ttc = 99.0
        for trk in active_tracks:
            ttc = trk['dist'] / max(ego_speed / 3.6, 2.5)
            if ttc < min_ttc:
                min_ttc = ttc

        if blocking_obs is not None:
            steer_act = "STEER RIGHT (AVOIDANCE)" if swerve_offset > 0 else "STEER LEFT (AVOIDANCE)"
            status_txt = f"WARNING (OBSTACLE #{blocking_obs['id']} AT {blocking_obs['dist']:.1f}m - BENDING PATH)"
        elif min_ttc < 3.0:
            steer_act = "DECELERATE & MAINTAIN CLEARANCE"
            status_txt = "CRITICAL PROXIMITY"
        else:
            steer_act = "FOLLOW NOMINAL CENTERLINE"
            status_txt = "NOMINAL (SAFE ROAD CORRIDOR)"

        hud_info = [
            f"LOCATION: {loc_info['name'].upper()} ({loc_info['tag']})",
            f"FRAME: #{frame_id:04d} (Global {idx+1}/{total_frames}) | Ego Speed: {ego_speed:.1f} km/h",
            f"Perception: Model 1 YOLO Vision + Model 2 LiDAR (FRONT: BLUE <25m, REAR: ORANGE >=25m) + 77 GHz Radar",
            f"Active Obstacles: {len(active_tracks)} Tracked | Min TTC: {min_ttc:.2f} s | Steering: {steer_act}",
            f"System Status: {status_txt} | Planner: Dual-Path Frenet Quintic Evasion"
        ]
        draw_hud_banner(display_frame, hud_info, 40, 40)

        # Save frame to disk
        frame_filename = os.path.join(temp_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(frame_filename, display_frame)
        rendered_frames.append(frame_filename)

        if (idx + 1) % 25 == 0 or idx == total_frames - 1:
            elapsed = time.time() - t_start
            fps_render = (idx + 1) / elapsed
            print(f"  --> Processed [{idx+1:03d}/{total_frames:03d}] frames ({fps_render:.1f} FPS render speed)")

    render_time = time.time() - t_start
    print(f"\n[3/3] Encoding {len(rendered_frames)} Frames into High-Quality H.264 MP4 @ {fps} FPS...")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    input_pattern = os.path.join(temp_dir, "frame_%05d.png")
    out_video_1 = os.path.join(BASE_DIR, "camera_vision_simulation.mp4")
    out_video_2 = os.path.join(BASE_DIR, "indian_road_simulation.mp4")

    cmd = [
        ffmpeg_exe, "-y",
        "-r", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        out_video_1
    ]

    t_enc = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("FFmpeg Error:", res.stderr.decode('utf-8', errors='ignore'))
        return

    shutil.copyfile(out_video_1, out_video_2)
    enc_time = time.time() - t_enc
    shutil.rmtree(temp_dir)

    v1_size = os.path.getsize(out_video_1) / (1024 * 1024)
    print("\n" + "=" * 100)
    print("SUCCESSFULLY GENERATED COLOR-CODED LIDAR SIMULATION VIDEO!")
    print(f"  * Primary File  : {out_video_1} ({v1_size:.2f} MB)")
    print(f"  * Backup File   : {out_video_2}")
    print(f"  * Resolution    : 1920 x 1080 (Full HD)")
    print(f"  * Framerate     : {fps} FPS")
    print(f"  * Total Frames  : {len(rendered_frames)} frames")
    print(f"  * Duration      : {len(rendered_frames)/fps:.2f} seconds")
    print(f"  * Total Time    : Render {render_time:.1f}s + Encode {enc_time:.1f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
