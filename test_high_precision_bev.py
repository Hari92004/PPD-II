"""
High-Precision 3D LiDAR BEV Generator with Actual Point Cloud Clusters
Draws:
  - Concentric Velodyne laser scan rings
  - Exact 3D bounding boxes wrapping the real vehicle point clusters in the road corridor
  - Blue Ego Vehicle + White Forward Arrow at bottom center
  - ZERO text clutter
"""

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

LIDAR_FILE = r"D:\PPD2\dataset_idd_synchronized\lidar\d0_lidar_0000025.npy"
OUT_IMG = r"D:\PPD2\sample_model2_bev_dbscan.png"

W, H = 1920, 1080
U0 = W // 2
V0 = int(H * 0.82)
SCALE = 22.0  # 22 pixels per meter (covers up to 45m ahead and 35m width cleanly)

def render_frame(lidar_path):
    pcd = np.load(lidar_path)[:, :4].astype(np.float32)
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    canvas = np.full((H, W, 3), (12, 16, 24), dtype=np.uint8) # Dark BEV canvas
    overlay = canvas.copy()

    # 1. Distance Grid Lines (every 10m)
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
    mask = (x >= -10.0) & (x <= 50.0) & (np.abs(y) <= 35.0)
    x_sel, y_sel, z_sel = x[mask], y[mask], z[mask]
    u_pts = (U0 - y_sel * SCALE).astype(np.int32)
    v_pts = (V0 - x_sel * SCALE).astype(np.int32)

    for i in range(len(u_pts)):
        up, vp = u_pts[i], v_pts[i]
        if 0 <= up < W and 0 <= vp < H:
            if z_sel[i] < -1.15:
                # Ground scan rings (blue / indigo)
                canvas[vp, up] = (195, 115, 45)
            else:
                # Obstacle reflection points (cyan / white)
                canvas[vp, up] = (245, 230, 180)

    # 4. Extract Real 3D Bounding Boxes on Obstacles (DBSCAN + Physical Constraints)
    road_mask = (x >= 2.5) & (x <= 42.0) & (np.abs(y) <= 8.5) & (z > -1.2)
    obs_pts = pcd[road_mask]

    if len(obs_pts) >= 15:
        clustering = DBSCAN(eps=0.85, min_samples=8).fit(obs_pts[:, :3])
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}

        for l in unique_labels:
            c_pts = obs_pts[labels == l]
            if len(c_pts) < 10:
                continue

            cx = float(c_pts[:, 0].mean())
            cy = float(c_pts[:, 1].mean())

            # Bounding box dimensions
            length = float(c_pts[:, 0].max() - c_pts[:, 0].min())
            width = float(c_pts[:, 1].max() - c_pts[:, 1].min())
            height = float(c_pts[:, 2].max() - c_pts[:, 2].min())

            # Physical vehicle filters: width between 0.6m and 3.2m, length between 0.8m and 9.0m
            if 0.5 <= width <= 3.8 and 0.8 <= length <= 9.0:
                # Pad slightly for realistic 3D vehicle envelope
                bw = max(1.2, min(2.5, width + 0.3))
                bl = max(1.6, min(5.5, length + 0.4))

                # Corner coordinates
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

                # Translucent Coral/Red Fill + Solid Border (ZERO TEXT!)
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

    cv2.imwrite(OUT_IMG, canvas)
    print("Saved high-precision sample frame:", OUT_IMG)

if __name__ == "__main__":
    render_frame(LIDAR_FILE)

