"""
Validation on Second Dataset Image (Drive 1 - Arterial Corridor)
Camera: images/d1_cam_0000076.jpg
LiDAR : lidar/d1_lidar_0000050.npy
"""

import os
import time
import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

BASE_DIR = r"D:\PPD2"
IDD_DIR = os.path.join(BASE_DIR, "dataset_idd_synchronized")
IMG_PATH = os.path.join(IDD_DIR, "images", "d1_cam_0000076.jpg")
LIDAR_PATH = os.path.join(IDD_DIR, "lidar", "d1_lidar_0000050.npy")
MODEL2_PATH = os.path.join(BASE_DIR, "model2", "model2_lidar.onnx")

print("=" * 90)
print("     VALIDATION ON SECOND IMAGE: DRIVE 1 (ARTERIAL CORRIDOR)")
print("=" * 90)

raw_img = cv2.imread(IMG_PATH)
H, W = raw_img.shape[:2]
print(f"  * Camera Image : {IMG_PATH} ({W}x{H})")
print(f"  * LiDAR File   : {LIDAR_PATH}")

# 1. Model 1 (2D Camera Vision YOLO)
yolo_model = YOLO("yolov8n.pt")
t0 = time.time()
results = yolo_model(raw_img, conf=0.25, verbose=False)
m1_latency = (time.time() - t0) * 1000

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
            'bbox': (x1, y1, x2 - x1, y2 - y1)
        })

print(f"\n[MODEL 1 CAMERA] Latency: {m1_latency:.2f} ms | Found {len(m1_detections)} Objects:")
for i, d in enumerate(m1_detections):
    bx, by, bw, bh = d['bbox']
    print(f"  [{i+1:02d}] Class: {d['class']:<12} | Conf: {d['conf']:.2%} | 2D BBox: [x={bx}, y={by}, w={bw}, h={bh}]")

# 2. Model 2 (3D LiDAR PointPillars)
raw_pts = np.load(LIDAR_PATH)[:, :4].astype(np.float32)
if len(raw_pts) >= 16384:
    inp2 = raw_pts[:16384][np.newaxis, ...]
else:
    pad = np.zeros((16384 - len(raw_pts), 4), dtype=np.float32)
    inp2 = np.vstack([raw_pts, pad])[np.newaxis, ...]

sess2 = ort.InferenceSession(MODEL2_PATH, providers=['CPUExecutionProvider'])
t0 = time.time()
out2 = sess2.run(None, {'point_cloud': inp2})[0]
m2_latency = (time.time() - t0) * 1000

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
                    'projected_2d': (x1, y1, x2 - x1, y2 - y1)
                })

print(f"\n[MODEL 2 LIDAR] Latency: {m2_latency:.2f} ms | Found {len(m2_detections)} 3D Boxes:")
for i, d in enumerate(m2_detections):
    px, py, pz = d['pos_3d']
    bx, by, bw, bh = d['projected_2d']
    pos_type = "FRONT" if d['dist'] < 25.0 else "REAR"
    print(f"  [{i+1:02d}] {pos_type:<5} | Class: {d['class']:<10} | Dist: {d['dist']:5.1f}m | 3D: (X={px:5.1f}, Y={py:5.1f}, Z={pz:5.1f}) | 2D: [{bx}, {by}, {bw}, {bh}]")

# 3. Draw Output Visualization
vis_img = raw_img.copy()

# LiDAR Points (Turbo colormap)
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

# Model 1 Vision Detections (Bright Green)
for d in m1_detections:
    bx, by, bw, bh = d['bbox']
    cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2, cv2.LINE_AA)
    lbl = f"M1: {d['class'].upper()} {d['conf']:.2f}"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    cv2.rectangle(vis_img, (bx, by - 20), (bx + tw + 6, by), (0, 255, 0), -1)
    cv2.putText(vis_img, lbl, (bx + 3, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)

# Model 2 LiDAR Projections: Front = BLUE (255, 150, 20), Rear = ORANGE (0, 140, 255)
COLOR_FRONT_BLUE = (255, 150, 20)   # Vivid Electric Blue (BGR)
COLOR_REAR_ORANGE = (0, 140, 255)   # Vivid Bright Orange (BGR)

for d in m2_detections:
    bx, by, bw, bh = d['projected_2d']
    is_front = d['dist'] < 25.0
    col = COLOR_FRONT_BLUE if is_front else COLOR_REAR_ORANGE
    box_tag = "FRONT" if is_front else "REAR"

    # Draw Bounding Box (Thick 3px)
    cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), col, 3, cv2.LINE_AA)

    # Label Card Below Box
    lbl = f"M2 ({box_tag}): {d['class'].upper()} {d['dist']:.1f}m"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    cv2.rectangle(vis_img, (bx, by + bh), (bx + tw + 6, by + bh + 22), col, -1)
    cv2.putText(vis_img, lbl, (bx + 3, by + bh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

# Telemetry Banner
cv2.rectangle(vis_img, (30, 30), (1200, 150), (12, 16, 24), -1)
cv2.rectangle(vis_img, (30, 30), (1200, 150), (70, 80, 95), 2)
cv2.putText(vis_img, "SECOND IMAGE VALIDATION: DRIVE 1 (ARTERIAL HIGHWAY CORRIDOR)", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
cv2.putText(vis_img, f"Model 1 (2D Camera Vision): {len(m1_detections)} Objects [GREEN]", (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 120), 2)
cv2.putText(vis_img, f"Model 2 Front Vehicles (<25m): BLUE [BGR (255,150,20)]", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 150, 20), 2)
cv2.putText(vis_img, f"Model 2 Rear Vehicles (>=25m): ORANGE [BGR (0,140,255)]", (580, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 140, 255), 2)

out_res_img = os.path.join(BASE_DIR, "validation_second_image_result.png")
cv2.imwrite(out_res_img, vis_img)
print(f"\n[SAVED] Result Image: {out_res_img}")
print("=" * 90)

