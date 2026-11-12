"""
Test Script to render a single sample BEV LiDAR frame matching user's reference style:
  - Dark BEV Canvas
  - Concentric scan rings / LiDAR point cloud
  - Ego Vehicle blue box + white arrow at bottom center
  - Model 2 PointPillars 3D predicted bounding boxes (translucent red)
  - NO TEXT LABELS (clean points + boxes)
"""

import os
import cv2
import numpy as np
import onnxruntime as ort

LIDAR_FILE = r"D:\PPD2\dataset_idd_synchronized\lidar\d0_lidar_0000016.npy"
MODEL2_ONNX = r"D:\PPD2\model2\model2_lidar.onnx"
OUT_IMG = r"D:\PPD2\sample_model2_bev.png"

W, H = 1280, 960
U0 = W // 2
V0 = int(H * 0.82)
SCALE = 15.0  # 15 pixels per meter

def render_bev_frame(pts, boxes, conf_thresh=0.30):
    canvas = np.full((H, W, 3), (12, 16, 24), dtype=np.uint8) # Dark navy background
    overlay = canvas.copy()

    # 1. Subtle Background Grid (every 10m)
    for d_m in range(-10, 60, 10):
        v = int(V0 - d_m * SCALE)
        if 0 <= v < H:
            cv2.line(canvas, (0, v), (W, v), (24, 32, 45), 1)
    for y_m in range(-30, 40, 10):
        u = int(U0 - y_m * SCALE)
        if 0 <= u < W:
            cv2.line(canvas, (u, 0), (u, H), (24, 32, 45), 1)

    # 2. Concentric Range Rings (10m, 20m, 30m, 40m, 50m)
    for r_m in [10, 20, 30, 40, 50]:
        r_px = int(r_m * SCALE)
        cv2.circle(canvas, (U0, V0), r_px, (35, 45, 65), 1, cv2.LINE_AA)

    # 3. Draw LiDAR Points
    # x = forward, y = lateral (+left, -right), z = height
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # Filter points in view
    mask = (x >= -12.0) & (x <= 55.0) & (np.abs(y) <= 35.0)
    x_sel, y_sel, z_sel = x[mask], y[mask], z[mask]

    u_pts = (U0 - y_sel * SCALE).astype(np.int32)
    v_pts = (V0 - x_sel * SCALE).astype(np.int32)

    # Color code: ground/near rings in cool blue/cyan, obstacles in brighter cyan/white
    dists = np.sqrt(x_sel**2 + y_sel**2)
    for i in range(len(u_pts)):
        up, vp = u_pts[i], v_pts[i]
        if 0 <= up < W and 0 <= vp < H:
            if z_sel[i] < -1.2:
                # Ground ring points (deep blue / indigo)
                canvas[vp, up] = (200, 120, 50)
            else:
                # Obstacle / object points (glowing cyan / white)
                canvas[vp, up] = (240, 230, 180)

    # 4. Draw Model 2 3D Bounding Boxes (Translucent Red / Orange, NO TEXT!)
    for b in boxes:
        conf = float(b[11])
        if conf >= conf_thresh:
            bx, by, bz = float(b[0]), float(b[1]), float(b[2])
            bw, bl, bh = float(b[3]), float(b[4]), float(b[5])
            sin_y, cos_y = float(b[6]), float(b[7])
            yaw = np.arctan2(sin_y, cos_y)

            # 4 corners in object coordinate frame (BEV)
            dx = bl / 2.0
            dy = bw / 2.0
            local_corners = np.array([
                [ dx,  dy],
                [ dx, -dy],
                [-dx, -dy],
                [-dx,  dy]
            ])

            # Rotate by yaw
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])
            rot_corners = local_corners @ R.T
            world_corners = rot_corners + np.array([bx, by])

            # Convert to canvas pixels
            poly_u = (U0 - world_corners[:, 1] * SCALE).astype(np.int32)
            poly_v = (V0 - world_corners[:, 0] * SCALE).astype(np.int32)
            poly_pts = np.column_stack([poly_u, poly_v]).reshape((-1, 1, 2))

            # Translucent Red/Coral Fill (matches user reference image!)
            cv2.fillPoly(overlay, [poly_pts], (60, 60, 230))
            cv2.polylines(canvas, [poly_pts], True, (100, 100, 255), 2, cv2.LINE_AA)

    # Blend translucent box fills
    cv2.addWeighted(overlay, 0.40, canvas, 0.60, 0, canvas)

    # 5. Draw Ego Vehicle (Blue Box + White Arrow, matching reference!)
    ego_w = int(2.0 * SCALE)   # ~30 px
    ego_l = int(4.5 * SCALE)   # ~67 px
    ego_x1 = U0 - ego_w // 2
    ego_x2 = U0 + ego_w // 2
    ego_y1 = V0 - ego_l // 2
    ego_y2 = V0 + ego_l // 2

    cv2.rectangle(canvas, (ego_x1, ego_y1), (ego_x2, ego_y2), (230, 140, 20), -1) # Blue fill
    cv2.rectangle(canvas, (ego_x1, ego_y1), (ego_x2, ego_y2), (255, 200, 80), 2)  # Bright cyan/blue border

    # White Forward Arrow inside Ego Box
    arrow_pts = np.array([
        [U0, V0 - 16],
        [U0 - 10, V0 + 10],
        [U0 + 10, V0 + 10]
    ], np.int32)
    cv2.fillPoly(canvas, [arrow_pts], (255, 255, 255), cv2.LINE_AA)

    return canvas

def main():
    pts = np.load(LIDAR_FILE)[:, :4].astype(np.float32)
    if len(pts) >= 16384:
        inp = pts[:16384][np.newaxis, ...]
    else:
        pad = np.zeros((16384 - len(pts), 4), dtype=np.float32)
        inp = np.vstack([pts, pad])[np.newaxis, ...]

    sess = ort.InferenceSession(MODEL2_ONNX, providers=['CPUExecutionProvider'])
    out = sess.run(None, {'point_cloud': inp})[0]
    boxes = out[0]

    img = render_bev_frame(pts, boxes)
    cv2.imwrite(OUT_IMG, img)
    print(f"[DONE] Saved sample BEV frame to: {OUT_IMG}")

if __name__ == "__main__":
    main()

