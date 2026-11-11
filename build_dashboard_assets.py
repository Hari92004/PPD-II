"""
BUILD DASHBOARD ASSETS:
Processes synchronized IDD camera & LiDAR frames, extracts 77 GHz mmWave radar target telemetry,
saves camera frames into D:\PPD2\dashboard_frames\ and metadata into D:\PPD2\dashboard_data.js.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

BASE_DIR = r"D:\PPD2"
IDD_DIR = os.path.join(BASE_DIR, "dataset_idd_synchronized")
INDEX_CSV = os.path.join(IDD_DIR, "synchronized_index.csv")

MODEL2_ONNX = os.path.join(BASE_DIR, "model2", "model2_lidar.onnx")
MODEL3_ONNX = os.path.join(BASE_DIR, "Model3_MoE", "model3_moe_trajectory.onnx")

COLOR_FRONT_BLUE = (255, 150, 20)    # Vibrant Blue
COLOR_CAM_GREEN = (0, 255, 0)        # Camera Green

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

def project_ego_point(s, d_lat=0.0, z=-1.55, cx_lane=960.0, cy_lane=535.0, W=1920, H=1080):
    fx, fy = 1250.0, 1250.0
    u = int((-d_lat * fx / s) + cx_lane)
    v = int((1.90 * fy / s) + cy_lane)
    return (u, v)

def render_dual_paths(frame, active_obstacles, ego_speed_kmh, drive_id='d0'):
    H, W = frame.shape[:2]
    overlay = frame.copy()
    s_ranges = np.linspace(3.0, 36.0, 35)
    cx_lane = 825.0 if drive_id == 'd0' else 960.0
    cy_lane = 535.0
    fx, fy = 1250.0, 1250.0
    half_w = 0.75  # 1.5m vehicle driving corridor width

    # 1. Search for critical obstacles in the forward road corridor (x in [4.0, 24.0], |y| <= 2.2m)
    blocking_obs = None
    min_dist = 999.0
    for obs in active_obstacles:
        ox, oy, _ = obs['pos']
        if 4.0 <= ox <= 24.0 and abs(oy) <= 2.2:
            if ox < min_dist:
                min_dist = ox
                blocking_obs = obs

    # 2. Dynamic Frenet Quintic Evasion Setup
    is_evading = False
    d_evasion = 0.0
    s_obs = 12.0
    if blocking_obs is not None:
        ox, oy, _ = blocking_obs['pos']
        s_obs = ox
        is_evading = True
        # Swerve away from obstacle:
        # If obstacle is on right (oy <= 0), smoothly bend LEFT (d_evasion = +0.85m)
        # If obstacle is on left (oy > 0), smoothly bend RIGHT (d_evasion = -0.85m)
        d_evasion = 0.85 if oy <= 0.0 else -0.85

    poly_left = []
    poly_right = []
    center_pts = []

    for s in s_ranges:
        if not is_evading:
            d_lat = 0.0
        else:
            # Zone 1: Starting region (s <= 4.2m) is STRICTLY STRAIGHT (Seedha)
            if s <= 4.2:
                d_lat = 0.0
            # Zone 2: Smooth Frenet Quintic Evasion Bend approaching the obstacle
            elif s <= s_obs + 4.0:
                tau = np.clip((s - 4.2) / max(3.5, s_obs - 1.0), 0.0, 1.0)
                poly = 10.0 * (tau**3) - 15.0 * (tau**4) + 6.0 * (tau**5)
                d_lat = d_evasion * poly
            # Zone 3: Smooth return to lane center past the obstacle
            else:
                tau_ret = np.clip((s - (s_obs + 4.0)) / 9.0, 0.0, 1.0)
                poly_ret = 10.0 * (tau_ret**3) - 15.0 * (tau_ret**4) + 6.0 * (tau_ret**5)
                d_lat = d_evasion * (1.0 - poly_ret)

        # 3. Project to Image
        u_c = int((-d_lat * fx / s) + cx_lane)
        v_c = int((1.90 * fy / s) + cy_lane)

        u_l = int((-(d_lat + half_w) * fx / s) + cx_lane)
        u_r = int((-(d_lat - half_w) * fx / s) + cx_lane)

        if 0 <= u_c < W and H / 2.0 <= v_c < H:
            center_pts.append((u_c, v_c))
        if 0 <= u_l < W and H / 2.0 <= v_c < H:
            poly_left.append((u_l, v_c))
        if 0 <= u_r < W and H / 2.0 <= v_c < H:
            poly_right.append((u_r, v_c))

    # 4. Render Ground Corridor Ribbon (Green when clear, Amber/Cyan when actively evading)
    if len(poly_left) >= 4 and len(poly_right) >= 4:
        carpet_pts = np.array(poly_left + poly_right[::-1], np.int32)
        ribbon_col = (0, 200, 255) if is_evading else (0, 225, 110)
        edge_col = (0, 230, 255) if is_evading else (0, 255, 180)

        cv2.fillPoly(overlay, [carpet_pts], ribbon_col)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        pts_l = np.array(poly_left, np.int32).reshape((-1, 1, 2))
        pts_r = np.array(poly_right, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_l], False, edge_col, 2, cv2.LINE_AA)
        cv2.polylines(frame, [pts_r], False, edge_col, 2, cv2.LINE_AA)

        pts_c = np.array(center_pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_c], False, (255, 255, 255), 2, cv2.LINE_AA)
        for i in range(2, len(center_pts) - 1, 3):
            cv2.circle(frame, center_pts[i], 3, (0, 240, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, center_pts[i], 5, edge_col, 1, cv2.LINE_AA)

    swerve_offset = d_evasion if is_evading else 0.0
    return swerve_offset, blocking_obs


def main():
    print("=" * 80)
    print("  BUILDING ASSETS FOR INTERACTIVE AUTONOMOUS VEHICLE DASHBOARD")
    print("=" * 80)

    frames_dir = os.path.join(BASE_DIR, "dashboard_frames")
    os.makedirs(frames_dir, exist_ok=True)

    df_all = pd.read_csv(INDEX_CSV)
    total_frames = 150
    frames_per_drive = total_frames // 3
    drives_frames = []
    for d in ['d0', 'd1', 'd2']:
        sub = df_all[df_all['drive'] == d]
        start_idx = min(25, max(0, len(sub) - frames_per_drive))
        chunk = sub.iloc[start_idx:start_idx + frames_per_drive]
        drives_frames.append(chunk)

    selected_df = pd.concat(drives_frames, ignore_index=True)
    print(f"Sampled {len(selected_df)} frames across d0, d1, d2.")

    print("Loading models...")
    yolo_model = YOLO("yolov8n.pt")
    m2_sess = ort.InferenceSession(MODEL2_ONNX, providers=['CPUExecutionProvider'])
    m3_sess = ort.InferenceSession(MODEL3_ONNX, providers=['CPUExecutionProvider'])

    dashboard_records = []
    t0 = time.time()

    DRIVE_NAMES = {
        'd0': 'Drive 0: MG Road Urban Sector (Dense Traffic)',
        'd1': 'Drive 1: Outer Ring Arterial Corridor (High Flow)',
        'd2': 'Drive 2: Suburban Highway Bypass (Wide Corridor)'
    }

    for idx in range(len(selected_df)):
        row = selected_df.iloc[idx]
        drive_id = str(row['drive'])
        frame_id = str(row['frame_id'])
        ego_speed = float(row['vehicle_speed_kmh'])

        img_path = os.path.join(IDD_DIR, row['camera_file'])
        lidar_path = os.path.join(IDD_DIR, row['lidar_file'])

        raw_frame = cv2.imread(img_path)
        if raw_frame is None:
            continue
        H, W = raw_frame.shape[:2]
        pts = np.load(lidar_path)[:, :4].astype(np.float32)

        display_cam = raw_frame.copy()

        # 1. YOLOv8 2D Vision
        yolo_results = yolo_model(raw_frame, conf=0.25, verbose=False)
        cam_vision_dets = []
        for r in yolo_results:
            for box in r.boxes:
                cid = int(box.cls[0])
                cname = yolo_model.names[cid]
                conf = float(box.conf[0])
                if cname in ['car', 'motorcycle', 'bus', 'truck', 'person', 'bicycle']:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    mapped_cls = 'truck_bus' if cname in ['bus', 'truck'] else ('two_wheeler' if cname in ['motorcycle', 'bicycle'] else cname)
                    cam_vision_dets.append({
                        'class': mapped_cls,
                        'conf': conf,
                        'box': (x1, y1, x2 - x1, y2 - y1),
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })

        # 2. LiDAR 3D PointPillars
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
        # Physical road corridor lateral limits (eliminates off-road building walls & trees)
        y_corridor_max = 3.2 if drive_id == 'd0' else (4.8 if drive_id == 'd1' else 6.5)

        for b in out2[0]:
            if b[11] > 0.35:
                x, y, z = float(b[0]), float(b[1]), float(b[2])
                l, w, h = float(b[3]), float(b[4]), float(b[5])
                if 3.5 < x < 48.0 and abs(y) <= y_corridor_max:
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

                    matched_cam = False
                    for vd in cam_vision_dets:
                        vx, vy, vw, vh = vd['box']
                        if abs(u - (vx + vw // 2)) < 90 and abs(v - (vy + vh // 2)) < 90:
                            cname = vd['class']
                            matched_cam = True
                            break

                    # Keep if matched with camera vision OR directly in forward lane
                    if (matched_cam or abs(y) <= 1.8) and (x2 - x1) > 25 and (y2 - y1) > 20 and y1 < H - 40 and 0 <= u < W and 0 <= v < H:
                        raw_fused_detections.append({
                            'class': cname,
                            'pos': (x, y, z),
                            'box': (x1, y1, x2 - x1, y2 - y1),
                            'center_2d': (u, v),
                            'dist': float(np.sqrt(x**2 + y**2)),
                            'conf': float(b[11])
                        })

        for vd in cam_vision_dets:
            vx, vy, vw, vh = vd['box']
            already_covered = False
            for rfd in raw_fused_detections:
                rx, ry, rw, rh = rfd['box']
                if abs((vx + vw // 2) - (rx + rw // 2)) < 90:
                    already_covered = True
                    break
            if not already_covered and vw > 45 and vh > 45:
                est_dist = max(5.0, min(50.0, 1250.0 * 2.0 / vw))
                est_x = est_dist
                est_y = -((vx + vw // 2) - cx) * est_x / fx
                if abs(est_y) <= y_corridor_max + 0.8:
                    raw_fused_detections.append({
                        'class': vd['class'],
                        'pos': (est_x, est_y, -1.2),
                        'box': (vx, vy, vw, vh),
                        'center_2d': (vx + vw // 2, vy + vh // 2),
                        'dist': float(np.sqrt(est_x**2 + est_y**2)),
                        'conf': vd['conf']
                    })

        # Dual Path Planner (Straight / Seedha Lane Carpet)
        swerve_offset, blocking_obs = render_dual_paths(display_cam, raw_fused_detections, ego_speed, drive_id=drive_id)

        # LiDAR Point Cloud Depth Scatter (Turbo colormap)
        u_p, v_p, d_p = project_lidar_points_fast(pts, W, H)
        if len(u_p) > 0:
            norm_depth = np.clip((d_p - 2.0) / 45.0, 0.0, 1.0)
            colors_depth = (norm_depth * 255).astype(np.uint8)
            colormap = cv2.applyColorMap(colors_depth, cv2.COLORMAP_TURBO)
            for i in range(0, len(u_p), 2):
                b, g, r = colormap[i, 0]
                cv2.circle(display_cam, (u_p[i], v_p[i]), 2, (int(b), int(g), int(r)), -1)

        # Model 1 Vision Green 2D Boxes
        for vd in cam_vision_dets:
            vx, vy, vw, vh = vd['box']
            cv2.rectangle(display_cam, (vx, vy), (vx + vw, vy + vh), COLOR_CAM_GREEN, 2, cv2.LINE_AA)
            cam_lbl = f"{vd['class'].upper()}"
            (tw, th), _ = cv2.getTextSize(cam_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(display_cam, (vx, vy - th - 6), (vx + tw + 6, vy), COLOR_CAM_GREEN, -1)
            cv2.putText(display_cam, cam_lbl, (vx + 3, vy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Model 2 LiDAR 3D Boxes: ONLY Front Vehicles (< 25m) in Blue!
        for det in raw_fused_detections:
            dist = det['dist']
            if dist < 25.0:
                bx, by, bw, bh = det['box']
                cv2.rectangle(display_cam, (bx, by), (bx + bw, by + bh), COLOR_FRONT_BLUE, 3, cv2.LINE_AA)
                lidar_tag = f"FRONT: {det['class'].upper()} {dist:.1f}m"
                cv2.rectangle(display_cam, (bx, by + bh), (bx + len(lidar_tag)*9 + 8, by + bh + 22), COLOR_FRONT_BLUE, -1)
                cv2.putText(display_cam, lidar_tag, (bx + 4, by + bh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # Crop Left 1340 width driving corridor (clean windshield)
        cropped_view = display_cam[:, 290:1630]
        # Resize to 960 x 720 for fast web rendering
        web_frame = cv2.resize(cropped_view, (960, 720), interpolation=cv2.INTER_AREA)
        frame_rel_path = f"dashboard_frames/frame_{idx:04d}.jpg"
        frame_abs_path = os.path.join(BASE_DIR, frame_rel_path)
        cv2.imwrite(frame_abs_path, web_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # 3. Simulate 77 GHz mmWave FMCW Radar Targets (Ahead & Behind)
        radar_targets_meta = []
        closest_r = 999.0
        min_ttc = 99.0

        for r_i, det in enumerate(raw_fused_detections):
            ox, oy, oz = det['pos']
            dist = det['dist']
            if dist < closest_r:
                closest_r = dist
            
            # Doppler radial velocity
            if 'truck' in det['class']:
                v_target = max(18.0, ego_speed - 2.5) / 3.6
                rcs_db = 22.0
            elif 'two_wheeler' in det['class']:
                v_target = max(20.0, ego_speed + 3.0) / 3.6
                rcs_db = 4.5
            else:
                v_target = (ego_speed + np.random.uniform(2.0, 8.0)) / 3.6
                rcs_db = 14.5

            ego_v_ms = ego_speed / 3.6
            rel_vx = v_target - ego_v_ms
            los_x = ox / (dist + 1e-5)
            vr = float(los_x * rel_vx)

            # TTC
            if vr < -0.5 and ox > 1.0:
                ttc_val = ox / abs(vr)
                if ttc_val < min_ttc:
                    min_ttc = ttc_val

            radar_targets_meta.append({
                'id': r_i + 1,
                'class': det['class'].upper(),
                'x': float(round(ox, 2)),      # Longitudinal (ahead)
                'y': float(round(oy, 2)),      # Lateral (+left, -right)
                'dist': float(round(dist, 1)),
                'vr': float(round(vr, 2)),
                'rcs': float(round(rcs_db, 1)),
                'is_front': bool(dist < 25.0)
            })

        # Add simulated rear follower target if drive is arterial or highway
        if drive_id in ['d1', 'd2']:
            rear_dist = float(round(7.5 + (idx % 12) * 0.4, 1))
            rear_vr = float(round(-0.8 + 0.3 * np.sin(idx * 0.4), 2))
            radar_targets_meta.append({
                'id': 99,
                'class': 'CAR (REAR FOLLOWER)',
                'x': float(-rear_dist),        # Behind ego (-x)
                'y': float(round(-0.4 + 0.2 * np.cos(idx * 0.3), 2)),
                'dist': float(rear_dist),
                'vr': float(rear_vr),
                'rcs': 14.0,
                'is_front': False
            })

        steer_angle_deg = float(round(swerve_offset * -2.4, 2))
        decision_str = "EVASION SWERVE (FRENET QUINTIC)" if blocking_obs else "FOLLOW NOMINAL CENTERLINE"

        dashboard_records.append({
            'index': idx,
            'drive': drive_id,
            'drive_name': DRIVE_NAMES.get(drive_id, drive_id),
            'frame_id': frame_id,
            'image_file': frame_rel_path,
            'ego_speed_kmh': float(round(ego_speed, 1)),
            'closest_dist_m': float(round(closest_r, 1)) if closest_r < 900 else None,
            'min_ttc_s': float(round(min_ttc, 2)) if min_ttc < 90 else None,
            'decision': decision_str,
            'steer_deg': steer_angle_deg,
            'active_locks': len(radar_targets_meta),
            'radar_targets': radar_targets_meta
        })

        if (idx + 1) % 30 == 0 or idx == total_frames - 1:
            print(f"  Processed [{idx+1:03d}/{total_frames:03d}] frames...")

    # Write dashboard_data.js
    js_content = f"// Autonomous Vehicle Multi-Modal Dashboard Data (Auto-Generated)\nconst DASHBOARD_DATA = {json.dumps(dashboard_records, indent=2)};\n"
    js_path = os.path.join(BASE_DIR, "dashboard_data.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"\n[DONE] Saved {len(dashboard_records)} frames to {frames_dir}")
    print(f"[DONE] Saved dashboard data to {js_path} ({os.path.getsize(js_path)/(1024):.1f} KB)")
    print(f"Total time: {time.time() - t0:.1f} s")

if __name__ == "__main__":
    main()

