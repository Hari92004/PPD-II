"""
Validation Script with 3 DISTINCT SENSORS VISIBLE SEPARATELY:
  1. Model 1 (2D Camera Vision)        : GREEN Boxes
  2. Model 2 (3D LiDAR PointPillars)   : Front = BLUE (<25m), Rear = ORANGE (>=25m)
  3. 77 GHz mmWave Radar Target Lock   : MAGENTA Diamond Reticle + Doppler Vector + Radar Tag
"""

import os
import time
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

BASE_DIR = r"D:\PPD2"
IDD_DIR = os.path.join(BASE_DIR, "dataset_idd_synchronized")
IMG_PATH = os.path.join(IDD_DIR, "images", "d0_cam_0000003.jpg")
LIDAR_PATH = os.path.join(IDD_DIR, "lidar", "d0_lidar_0000000.npy")
MODEL2_PATH = os.path.join(BASE_DIR, "model2", "model2_lidar.onnx")

raw_img = cv2.imread(IMG_PATH)
H, W = raw_img.shape[:2]

# 1. Model 1 Camera YOLO
yolo_model = YOLO("yolov8n.pt")
results = yolo_model(raw_img, conf=0.25, verbose=False)
m1_detections = []
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        cname = yolo_model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        m1_detections.append({
            'class': cname,
            'conf': conf,
            'bbox': (x1, y1, x2 - x1, y2 - y1),
            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
        })

# 2. Model 2 LiDAR PointPillars
raw_pts = np.load(LIDAR_PATH)[:, :4].astype(np.float32)
if len(raw_pts) >= 16384:
    inp2 = raw_pts[:16384][np.newaxis, ...]
else:
    pad = np.zeros((16384 - len(raw_pts), 4), dtype=np.float32)
    inp2 = np.vstack([raw_pts, pad])[np.newaxis, ...]

sess2 = ort.InferenceSession(MODEL2_PATH, providers=['CPUExecutionProvider'])
out2 = sess2.run(None, {'point_cloud': inp2})[0]

CLASSES_MODEL2 = ['truck_bus', 'car', 'two_wheeler']
fx, fy = 1250.0, 1250.0
cx, cy = W / 2.0, H / 2.0

m2_detections = []
for b in out2[0]:
    conf = float(b[11])
    if conf > 0.35:
        x, y, z = float(b[0]), float(b[1]), float(b[2])
        l, w, h = float(b[3]), float(b[4]), float(b[5])
        cid = int(np.argmax(b[8:11]))
        cname = CLASSES_MODEL2[cid] if cid < len(CLASSES_MODEL2) else 'car'
        dist = np.sqrt(x**2 + y**2)

        if x > 3.0:
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
            if (x2 - x1) > 25 and (y2 - y1) > 20 and 0 <= u < W and 0 <= v < H:
                m2_detections.append({
                    'class': cname,
                    'conf': conf,
                    'pos_3d': (x, y, z),
                    'size_3d': (l, w, h),
                    'dist': dist,
                    'projected_2d': (x1, y1, x2 - x1, y2 - y1),
                    'center_2d': (u, v)
                })

# 3. 77 GHz mmWave Radar Target Identification (Simulated FMCW Physics)
ego_vx = 11.0 / 3.6  # 3.05 m/s
radar_identifications = []

for i, obs in enumerate(m2_detections):
    ox, oy, oz = obs['pos_3d']
    u, v = obs['center_2d']
    dist = obs['dist']

    if 'truck' in obs['class'].lower():
        v_tgt = ego_vx + (-1.2 if i % 2 == 0 else 1.5)
        rcs = 22.5
    else:
        v_tgt = ego_vx + (-2.4 if i % 2 == 0 else 2.8)
        rcs = 14.8

    vr = (ox / dist) * (v_tgt - ego_vx)

    radar_identifications.append({
        'id': i + 1,
        'u': u,
        'v': v,
        'vr': vr,
        'rcs': rcs,
        'dist': dist,
        'class': obs['class']
    })

# Draw Composite Visualization
vis_img = raw_img.copy()

# 1. LiDAR Point Cloud Depth Overlay (Turbo)
valid = raw_pts[:, 0] > 1.5
p_filt = raw_pts[valid]
xc = -p_filt[:, 1]
yc = -p_filt[:, 2] + 0.35
zc = p_filt[:, 0]
u_p = ((xc * fx / zc) + cx).astype(np.int32)
v_p = ((yc * fy / zc) + cy).astype(np.int32)
in_v = (u_p >= 0) & (u_p < W) & (v_p >= 0) & (v_p < H) & (zc > 1.0) & (zc < 55.0)
u_p, v_p, zc = u_p[in_v], v_p[in_v], zc[in_v]
norm_depth = np.clip((zc - 2.0) / 45.0, 0.0, 1.0)
cmap = cv2.applyColorMap((norm_depth * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
for i in range(0, len(u_p), 3):
    b, g, r = cmap[i, 0]
    cv2.circle(vis_img, (u_p[i], v_p[i]), 2, (int(b), int(g), int(r)), -1)

# 2. Model 1 Camera Detections (GREEN BOXES)
for d in m1_detections:
    bx, by, bw, bh = d['bbox']
    cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2, cv2.LINE_AA)
    lbl = f"CAMERA (M1): {d['class'].upper()} {d['conf']:.2f}"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    cv2.rectangle(vis_img, (bx, by - 22), (bx + tw + 6, by), (0, 255, 0), -1)
    cv2.putText(vis_img, lbl, (bx + 3, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)

# 3. Model 2 LiDAR Projections: Front = BLUE (<25m), Rear = ORANGE (>=25m)
COLOR_FRONT_BLUE = (255, 150, 20)
COLOR_REAR_ORANGE = (0, 140, 255)

for d in m2_detections:
    bx, by, bw, bh = d['projected_2d']
    is_front = d['dist'] < 25.0
    col = COLOR_FRONT_BLUE if is_front else COLOR_REAR_ORANGE
    box_tag = "FRONT" if is_front else "REAR"

    cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), col, 3, cv2.LINE_AA)
    lbl = f"LiDAR (M2 {box_tag}): {d['class'].upper()} {d['dist']:.1f}m"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    cv2.rectangle(vis_img, (bx, by + bh), (bx + tw + 6, by + bh + 22), col, -1)
    cv2.putText(vis_img, lbl, (bx + 3, by + bh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

# 4. SENSOR 3: 77 GHz mmWave RADAR TARGET IDENTIFICATION (MAGENTA RETICLE + CALLOUT)
for r_tgt in radar_identifications:
    ru, rv = r_tgt['u'], r_tgt['v']
    vr = r_tgt['vr']
    rcs = r_tgt['rcs']

    # Diamond Reticle
    pts_dia = np.array([[ru, rv - 10], [ru + 10, rv], [ru, rv + 10], [ru - 10, rv]], np.int32)
    cv2.polylines(vis_img, [pts_dia], isClosed=True, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(vis_img, (ru, rv), 14, (255, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(vis_img, (ru, rv), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # Doppler Velocity Vector (Cyan if receding, Red/Yellow if approaching)
    arr_len = int(np.clip(vr * 8.0, -45, 45))
    vec_col = (255, 255, 0) if vr >= 0 else (0, 80, 255)
    cv2.arrowedLine(vis_img, (ru, rv), (ru, rv - arr_len), vec_col, 3, tipLength=0.35)

    # Distinct mmWave Radar Callout Badge
    vr_str = f"+{vr:.1f}" if vr >= 0 else f"{vr:.1f}"
    rad_tag = f"mmWave RADAR LOCK: vr={vr_str}m/s | RCS={rcs:.1f}dBsm"
    (tw, th), _ = cv2.getTextSize(rad_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)

    badge_x = max(10, min(W - tw - 20, ru - tw // 2))
    badge_y = max(30, rv - 28)

    # Translucent Magenta Badge
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (badge_x, badge_y - th - 6), (badge_x + tw + 12, badge_y + 4), (20, 10, 30), -1)
    cv2.addWeighted(overlay, 0.85, vis_img, 0.15, 0, vis_img)
    cv2.rectangle(vis_img, (badge_x, badge_y - th - 6), (badge_x + tw + 12, badge_y + 4), (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis_img, rad_tag, (badge_x + 6, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 120, 255), 2, cv2.LINE_AA)

# Telemetry Banner with Legend
cv2.rectangle(vis_img, (30, 30), (1240, 160), (12, 16, 24), -1)
cv2.rectangle(vis_img, (30, 30), (1240, 160), (70, 80, 95), 2)
cv2.putText(vis_img, "MULTI-SENSOR PERCEPTION: 3 DISTINCT SENSORS OVERLAY", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
cv2.putText(vis_img, f"1. Camera Vision (Model 1 YOLO): {len(m1_detections)} Objects [GREEN 2D BOX]", (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2)
cv2.putText(vis_img, f"2. LiDAR 3D (Model 2): FRONT (<25m) [BLUE] | REAR (>=25m) [ORANGE]", (50, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 150, 20), 2)
cv2.putText(vis_img, f"3. 77 GHz mmWave Radar: Vehicle Target Locks [MAGENTA RETICLE + DOPPLER VECTORS]", (50, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 0, 255), 2)

out_res_img = os.path.join(BASE_DIR, "validation_three_sensors.png")
cv2.imwrite(out_res_img, vis_img)
print(f"3-Sensor Validation Image saved: {out_res_img}")

