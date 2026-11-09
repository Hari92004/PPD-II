"""
BEV (Bird's Eye View) LiDAR 3D Detection & 77 GHz mmWave Radar Point-Out Simulation
For Second Image: Drive 1 Arterial Corridor (d1_lidar_0000050.npy)
Style: Matches dark-theme polar-cartesian BEV grid with Doppler vectors and RCS callouts.
Output: D:\PPD2\lidar_mmwave_bev_simulation.png
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import onnxruntime as ort

BASE_DIR = r"D:\PPD2"
IDD_DIR = os.path.join(BASE_DIR, "dataset_idd_synchronized")
LIDAR_PATH = os.path.join(IDD_DIR, "lidar", "d1_lidar_0000050.npy")
MODEL2_PATH = os.path.join(BASE_DIR, "model2", "model2_lidar.onnx")

print("=" * 80)
print("  GENERATING BEV LIDAR & 77 GHz mmWave RADAR FOR DRIVE 1 (d1_lidar_0000050)")
print("=" * 80)

# 1. Load LiDAR points
pts_raw = np.load(LIDAR_PATH)[:, :4].astype(np.float32)
print(f"Loaded {len(pts_raw)} LiDAR points.")

# 2. Model 2 3D Inference
if len(pts_raw) >= 16384:
    inp2 = pts_raw[:16384][np.newaxis, ...]
else:
    pad = np.zeros((16384 - len(pts_raw), 4), dtype=np.float32)
    inp2 = np.vstack([pts_raw, pad])[np.newaxis, ...]

sess2 = ort.InferenceSession(MODEL2_PATH, providers=['CPUExecutionProvider'])
t0 = time.time()
out2 = sess2.run(None, {'point_cloud': inp2})[0]
latency_ms = (time.time() - t0) * 1000

CLASSES_MODEL2 = ['Truck/Bus', 'Car', 'Two-Wheeler']
detected_boxes = []

for b in out2[0]:
    conf = float(b[11])
    if conf > 0.35:
        x, y, z = float(b[0]), float(b[1]), float(b[2])
        l, w, h = float(b[3]), float(b[4]), float(b[5])
        cid = int(np.argmax(b[8:11]))
        cname = CLASSES_MODEL2[cid] if cid < len(CLASSES_MODEL2) else 'Car'
        dist = np.sqrt(x**2 + y**2)
        if 4.0 < x < 55.0 and abs(y) < 35.0:
            detected_boxes.append({
                'class': cname,
                'conf': conf,
                'x': x, 'y': y, 'z': z,
                'l': l, 'w': w, 'h': h,
                'dist': dist
            })

print(f"Model 2 detected {len(detected_boxes)} 3D obstacles in {latency_ms:.2f} ms.")

# 3. Simulate 77 GHz FMCW Radar Targets Pointing Out Each Vehicle
radar_targets = []
ego_vx = 18.0 / 3.6  # 18 km/h = 5.0 m/s

for obs in detected_boxes:
    ox, oy = obs['x'], obs['y']
    dist = obs['dist']
    if 'Truck' in obs['class']:
        v_tgt = ego_vx + np.random.uniform(-2.0, 2.0)
        rcs_db = np.random.uniform(19.0, 24.5)
    else:
        v_tgt = ego_vx + np.random.uniform(1.5, 4.5)
        rcs_db = np.random.uniform(11.0, 15.0)

    los_x = ox / dist
    los_y = oy / dist
    v_rel_x = v_tgt - ego_vx
    vr_doppler = los_x * v_rel_x

    radar_targets.append({
        'x': ox,
        'y': oy,
        'vr': vr_doppler,
        'rcs': rcs_db,
        'dist': dist,
        'parent_obs': obs
    })

print(f"Generated {len(radar_targets)} 77 GHz mmWave radar reflections pointing out vehicles.")

# 4. Create High-Tech Dark Theme BEV Figure
fig, ax = plt.subplots(figsize=(12, 12), dpi=150, facecolor='#0b0e14')
ax.set_facecolor('#0b0e14')

# Polar Range Rings (10m, 20m, 30m, 40m, 50m)
theta_grid = np.linspace(-np.pi/2, np.pi/2, 200)
for r in [10, 20, 30, 40, 50]:
    rx = r * np.cos(theta_grid)
    ry = r * np.sin(theta_grid)
    ax.plot(ry, rx, color='#212d3d', linestyle='--', linewidth=0.9, alpha=0.85)
    ax.text(0.5, r + 0.5, f'{r}m', color='#4f637a', fontsize=9, fontweight='bold', ha='center', va='bottom')

# Radial Sector Rays
for az in [-45, -30, -15, 0, 15, 30, 45]:
    rad = np.radians(az)
    ax.plot([0, 55 * np.sin(rad)], [0, 55 * np.cos(rad)], color='#182230', linestyle=':', linewidth=0.8)

# Subtle Cartesian Grid
ax.grid(color='#182230', linestyle='-', linewidth=0.6, alpha=0.7)

# Filter LiDAR points for BEV display
fov_mask = (pts_raw[:, 0] > -5.0) & (pts_raw[:, 0] < 55.0) & (np.abs(pts_raw[:, 1]) < 38.0) & (pts_raw[:, 2] > -2.8) & (pts_raw[:, 2] < 3.0)
pts_bev = pts_raw[fov_mask]

# Scatter LiDAR Points (Intensity/Depth colormap)
z_vals = pts_bev[:, 2]
scatter = ax.scatter(pts_bev[:, 1], pts_bev[:, 0], c=z_vals, cmap='viridis', s=1.2, alpha=0.55, rasterized=True)

# Plot Ego Vehicle at Origin (0, 0)
ax.plot(0, 0, marker='^', markersize=14, color='#00e5ff', markeredgecolor='#ffffff', markeredgewidth=1.5, zorder=10)
ax.text(0, -3.2, 'EGO VEHICLE', color='#00e5ff', fontsize=10, fontweight='bold', ha='center', va='top')

# Model 2 3D Bounding Boxes: Front (< 25m) = BLUE, Rear (>= 25m) = ORANGE
COLOR_FRONT_BLUE = '#00a6ff'    # Blue
COLOR_REAR_ORANGE = '#ff7700'   # Orange

for obs in detected_boxes:
    ox, oy = obs['x'], obs['y']
    l, w = obs['l'], obs['w']
    dist = obs['dist']
    is_front = dist < 25.0
    col = COLOR_FRONT_BLUE if is_front else COLOR_REAR_ORANGE
    box_tag = "FRONT" if is_front else "REAR"

    # Oriented bounding box
    rect = patches.Rectangle((oy - w/2.0, ox - l/2.0), w, l,
                             linewidth=2.0, edgecolor=col, facecolor=col, alpha=0.18, zorder=5)
    ax.add_patch(rect)
    rect_border = patches.Rectangle((oy - w/2.0, ox - l/2.0), w, l,
                                    linewidth=1.8, edgecolor=col, facecolor='none', zorder=6)
    ax.add_patch(rect_border)

    # Box Label Tag
    lbl = f"{box_tag} {obs['class'].upper()}\n{dist:.1f}m ({obs['conf']:.2f})"
    ax.text(oy, ox + l/2.0 + 0.8, lbl, color=col, fontsize=7.5, fontweight='bold',
            ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.25', facecolor='#0b0e14', edgecolor=col, alpha=0.85, linewidth=1.0), zorder=8)

# --------------------------------------------------------------------------------------------------
# POINT OUT VEHICLES WITH 77 GHz mmWave RADAR TARGETS
# --------------------------------------------------------------------------------------------------
for rt in radar_targets:
    rx, ry = rt['x'], rt['y']
    vr = rt['vr']
    rcs = rt['rcs']

    # 1. Radar Diamond Marker (Magenta Glowing Diamond)
    ax.plot(ry, rx, marker='D', markersize=9, color='#ff00ff', markeredgecolor='#ffffff', markeredgewidth=1.2, zorder=9)

    # 2. Doppler Velocity Vector Arrow (Red if approaching, Cyan if receding)
    arr_len = np.clip(vr * 1.8, -4.5, 4.5)
    arr_col = '#00ffff' if vr >= 0 else '#ff3366'
    ax.annotate('', xy=(ry + arr_len * (ry / (rt['dist'] + 1e-6)), rx + arr_len * (rx / (rt['dist'] + 1e-6))),
                xytext=(ry, rx),
                arrowprops=dict(arrowstyle="->", color=arr_col, lw=2.2, mutation_scale=14),
                zorder=9)

    # 3. Radar Point-Out Callout Tag
    vr_sign = "+" if vr >= 0 else ""
    callout_txt = f"RADAR POINT-OUT\nvr: {vr_sign}{vr:.1f} m/s\n\u03c3: {rcs:.1f} dBsm"
    offset_x = 5.0 if ry >= 0 else -5.0
    ax.annotate(callout_txt, xy=(ry, rx), xytext=(ry + offset_x, rx + 1.5),
                arrowprops=dict(arrowstyle="-|>", color='#ff00ff', lw=1.2, linestyle='--'),
                fontsize=7.0, color='#ff77ff', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#150d1e', edgecolor='#ff00ff', alpha=0.90, linewidth=1.2),
                zorder=10)

# Limits & Labels
ax.set_xlim(-38, 38)
ax.set_ylim(-6, 56)
ax.set_xlabel("Lateral Distance Y (m)", fontsize=12, fontweight='bold', color='#c9d1d9', labelpad=10)
ax.set_ylabel("Longitudinal Distance X (m)", fontsize=12, fontweight='bold', color='#c9d1d9', labelpad=10)
ax.tick_params(colors='#8b949e', labelsize=10)

# Title & Subtitle Banner
plt.title(f"Model 2: 3D LiDAR Object Detection (PointPillars BEV) + 77 GHz mmWave Radar Point-Out\nDrive 1 Arterial Corridor | Latency: {latency_ms:.2f} ms | Detected Obstacles: {len(detected_boxes)}",
          fontsize=12.5, fontweight='bold', color='#ffffff', pad=15)

# Custom Legend
legend_elements = [
    Line2D([0], [0], marker='^', color='none', markerfacecolor='#00e5ff', markeredgecolor='#ffffff', markersize=11, label='Ego Vehicle (Origin)'),
    Line2D([0], [0], color=COLOR_FRONT_BLUE, lw=2.5, label='Front Vehicle (< 25m, BLUE)'),
    Line2D([0], [0], color=COLOR_REAR_ORANGE, lw=2.5, label='Rear Vehicle (>= 25m, ORANGE)'),
    Line2D([0], [0], marker='D', color='none', markerfacecolor='#ff00ff', markeredgecolor='#ffffff', markersize=9, label='77 GHz mmWave Radar Target Point-Out (\u03c3 in dBsm)'),
    Line2D([0], [0], color='#00ffff', lw=2.0, label='Doppler Velocity Vector (\u2192 Receding / Moving Away)'),
    Line2D([0], [0], color='#ff3366', lw=2.0, label='Doppler Velocity Vector (\u2192 Approaching / Inbound)')
]
leg = ax.legend(handles=legend_elements, loc='upper right', facecolor='#111622', edgecolor='#30363d', fontsize=9, framealpha=0.92)
for text in leg.get_texts():
    text.set_color('#e6edf3')

plt.tight_layout()
out_file = os.path.join(BASE_DIR, "lidar_mmwave_bev_simulation.png")
plt.savefig(out_file, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"\n[SUCCESS] BEV LiDAR + mmWave Radar Simulation saved to: {out_file}")
print(f"  * Resolution: 2160 x 2160 High-Res PNG ({os.path.getsize(out_file)/(1024*1024):.2f} MB)")
print("=" * 80)
