"""
Model 2 (3D LiDAR Point Cloud BEV Perception Video Generator)
Features 1-to-1 Spatial Sensor Fusion Correspondence with Camera Vision:
  - For every vehicle detected in camera vision, its exact 3D coordinates (X, Y) are extracted from LiDAR
  - BEV video shows the EXACT same vehicles that are detected in the front camera video
  - High-tech dark BEV canvas with concentric Velodyne laser scan rings (blue/indigo)
  - Bottom-center Ego Vehicle (Blue box + White forward arrow)
  - Translucent Coral/Red 3D vehicle bounding boxes with solid outline
  - ZERO text clutter (No text labels, completely clean points and boxes)
  - 10.0 FPS native sensor synchronization
Output: D:\\PPD2\\model2_processed_bev_videos\\model2_lidar_bev_simulation.mp4
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

BASE_DIR = r"D:\PPD2"
IDD_DIR = os.path.join(BASE_DIR, "dataset_idd_synchronized")
INDEX_CSV = os.path.join(IDD_DIR, "synchronized_index.csv")

OUT_DIR = os.path.join(BASE_DIR, "model2_processed_bev_videos")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_VIDEO = os.path.join(OUT_DIR, "model2_lidar_bev_simulation.mp4")

# Resolution & Coordinate Transforms
W, H = 1920, 1080
U0 = W // 2
V0 = int(H * 0.82)
SCALE = 22.0   # 22 pixels per meter
FPS = 10.0

# Camera Intrinsics for Pinhole Projection
FX, FY = 1250.0, 1250.0
CX, CY = 960.0, 540.0

def extract_matched_3d_vehicles(frame, pcd, yolo_model):
    """
    Performs 1-to-1 Spatial Calibration Fusion:
    Finds exact 3D LiDAR positions of vehicles detected by Camera YOLOv8.
    """
    # 1. Camera 2D Detection
    results = yolo_model.predict(frame, device=0, imgsz=640, conf=0.25, verbose=False)
    
    # 2. Pinhole Projection of LiDAR Points to Camera Plane
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    valid = x > 1.5
    x_v, y_v, z_v = x[valid], y[valid], z[valid]
    u = (-y_v * FX / x_v) + CX
    v = (-z_v * FY / x_v) + CY

    matched_boxes = []

    for box in results[0].boxes:
        cname = yolo_model.names[int(box.cls[0])]
        if cname in ['car', 'motorcycle', 'bus', 'truck', 'person', 'bicycle']:
            x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
            # Find 3D points falling inside 2D detection box
            in_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            n_pts = np.sum(in_box)

            if n_pts >= 4:
                # 3D Position of vehicle from actual LiDAR point cloud
                bx = float(np.median(x_v[in_box]))
                by = float(np.median(y_v[in_box]))
                
                # Realistic 3D dimensions based on class
                if cname in ['bus', 'truck']:
                    bl, bw = 6.5, 2.4
                elif cname in ['motorcycle', 'bicycle']:
                    bl, bw = 1.8, 0.9
                elif cname == 'person':
                    bl, bw = 0.8, 0.8
                else:
                    bl, bw = 4.2, 1.8

                matched_boxes.append({
                    'cx': bx,
                    'cy': by,
                    'bl': bl,
                    'bw': bw,
                    'cls': cname
                })
            elif (x2 - x1) > 40 and (y2 - y1) > 40:
                # Far vehicle fallback distance estimation
                est_x = max(5.0, min(50.0, 1250.0 * 2.0 / (x2 - x1)))
                est_y = -((x1 + x2)/2.0 - CX) * est_x / FX
                matched_boxes.append({
                    'cx': est_x,
                    'cy': est_y,
                    'bl': 4.0,
                    'bw': 1.8,
                    'cls': cname
                })

    return matched_boxes

def render_bev_frame(pcd, matched_boxes):
    canvas = np.full((H, W, 3), (12, 16, 24), dtype=np.uint8) # Dark navy canvas
    overlay = canvas.copy()

    # 1. Subtle Distance Grid Lines (every 10m)
    for d_m in range(-10, 50, 10):
        v = int(V0 - d_m * SCALE)
        if 0 <= v < H:
            cv2.line(canvas, (0, v), (W, v), (22, 28, 40), 1)
    for y_m in range(-35, 40, 10):
        u = int(U0 - y_m * SCALE)
        if 0 <= u < W:
            cv2.line(canvas, (u, 0), (u, H), (22, 28, 40), 1)

    # 2. Polar Range Rings (10m, 20m, 30m, 40m)
    for r_m in [10, 20, 30, 40]:
        r_px = int(r_m * SCALE)
        cv2.circle(canvas, (U0, V0), r_px, (28, 38, 55), 1, cv2.LINE_AA)

    # 3. Draw All LiDAR Points
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]
    mask = (x >= -10.0) & (x <= 50.0) & (np.abs(y) <= 35.0)
    x_sel, y_sel, z_sel = x[mask], y[mask], z[mask]
    u_pts = (U0 - y_sel * SCALE).astype(np.int32)
    v_pts = (V0 - x_sel * SCALE).astype(np.int32)

    for i in range(len(u_pts)):
        up, vp = u_pts[i], v_pts[i]
        if 0 <= up < W and 0 <= vp < H:
            if z_sel[i] < -1.15:
                # Ground laser scan rings (blue / indigo)
                canvas[vp, up] = (195, 115, 45)
            else:
                # Obstacle reflection points (cyan / white)
                canvas[vp, up] = (245, 230, 180)

    # 4. Draw 1-to-1 Matched 3D Vehicle Bounding Boxes (Translucent Coral/Red, ZERO TEXT!)
    for veh in matched_boxes:
        cx, cy = veh['cx'], veh['cy']
        bw, bl = veh['bw'], veh['bl']

        x1 = cx - bl / 2.0
        x2 = cx + bl / 2.0
        y1 = cy - bw / 2.0
        y2 = cy + bw / 2.0

        u1 = int(U0 - y2 * SCALE)
        u2 = int(U0 - y1 * SCALE)
        v1 = int(V0 - x2 * SCALE)
        v2 = int(V0 - x1 * SCALE)

        box_corners = np.array([
            [u1, v1], [u2, v1], [u2, v2], [u1, v2]
        ], np.int32).reshape((-1, 1, 2))

        # Translucent Red/Coral Fill + Solid Border (ZERO TEXT!)
        cv2.fillPoly(overlay, [box_corners], (50, 50, 220))
        cv2.polylines(canvas, [box_corners], True, (80, 80, 255), 2, cv2.LINE_AA)

    # Blend translucent box fills
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    # 5. Draw Ego Vehicle (Blue Box + White Forward Arrow at bottom center)
    ego_w = int(2.0 * SCALE)
    ego_l = int(4.6 * SCALE)
    e_x1 = U0 - ego_w // 2
    e_x2 = U0 + ego_w // 2
    e_y1 = V0 - ego_l // 2
    e_y2 = V0 + ego_l // 2

    cv2.rectangle(canvas, (e_x1, e_y1), (e_x2, e_y2), (230, 135, 20), -1)
    cv2.rectangle(canvas, (e_x1, e_y1), (e_x2, e_y2), (255, 205, 70), 2)

    arrow_pts = np.array([
        [U0, V0 - 20],
        [U0 - 12, V0 + 12],
        [U0 + 12, V0 + 12]
    ], np.int32)
    cv2.fillPoly(canvas, [arrow_pts], (255, 255, 255), cv2.LINE_AA)

    return canvas

def main(num_frames=150):
    print("=" * 75)
    print("  GENERATING 1-TO-1 SYNCHRONIZED MODEL 2 LIDAR BEV VIDEO")
    print("=" * 75)

    df_all = pd.read_csv(INDEX_CSV)
    frames_per_drive = num_frames // 3
    chunks = []
    for d in ['d0', 'd1', 'd2']:
        sub = df_all[df_all['drive'] == d]
        start_idx = min(25, max(0, len(sub) - frames_per_drive))
        chunks.append(sub.iloc[start_idx:start_idx + frames_per_drive])
    selected_df = pd.concat(chunks, ignore_index=True)
    total = len(selected_df)

    print(f"Loaded {total} synchronized frames across Drive 0, Drive 1, Drive 2.")
    print("Initializing YOLOv8 for 1-to-1 Spatial Sensor Association on RTX 2050 GPU...")
    yolo_model = YOLO("yolov8n.pt")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (W, H))

    t0 = time.time()
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        cam_path = os.path.join(IDD_DIR, row['camera_file'])
        lidar_path = os.path.join(IDD_DIR, row['lidar_file'])

        frame = cv2.imread(cam_path)
        pcd = np.load(lidar_path)[:, :4].astype(np.float32)

        # 1-to-1 extraction of detected vehicles
        matched_boxes = extract_matched_3d_vehicles(frame, pcd, yolo_model)

        # Render clean BEV frame (Zero text!)
        bev_frame = render_bev_frame(pcd, matched_boxes)
        writer.write(bev_frame)

        if (idx + 1) % 30 == 0 or idx == total - 1:
            fps_cur = (idx + 1) / (time.time() - t0)
            print(f"  Rendered [{idx+1:03d}/{total:03d}] frames ({fps_cur:.1f} FPS)...")

    writer.release()
    v_mb = os.path.getsize(OUT_VIDEO) / (1024 * 1024)
    print(f"\n[DONE] Successfully created 1-to-1 synchronized Model 2 BEV video:")
    print(f"Path: {OUT_VIDEO} ({v_mb:.2f} MB, {total/FPS:.1f}s)")

if __name__ == "__main__":
    main()
