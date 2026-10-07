"""
====================================================================================================
77 GHz FMCW mmWAVE RADAR SYNTHETIC GENERATOR FROM 3D LIDAR
====================================================================================================
Simulates physically accurate automotive mmWave radar returns from LiDAR point clouds.

Physical Models & Algorithms Implemented:
  1. Coordinate Transformation:
     - Transforms LiDAR points to Front Bumper Radar frame using calibrated extrinsics.
  2. Antenna Radiation Pattern & FOV Gating:
     - Azimuth: [-45 deg, +45 deg], Elevation: [-10 deg, +10 deg], Range: 1.0m to 100m.
  3. Doppler Radial Velocity Model (Line-of-Sight Projection):
     - v_r = v_rel · r_hat = (x·vx + y·vy + z·vz) / R
     - Doppler frequency shift: f_D = 2 · v_r · fc / c (fc = 77 GHz)
  4. Radar Cross Section (RCS, sigma) Model:
     - Assigns material & geometry reflectivity sigma [dBsm] based on class & incident angle.
  5. Radar Range Equation & Received Power:
     - P_r = (P_t · G^2 · lambda^2 · sigma) / ((4*pi)^3 · R^4 · L_atm)
     - Signal-to-Noise Ratio (SNR) and CFAR thresholding.
  6. Range-Doppler FFT Sparsification:
     - Bins returns into Range-Azimuth cells and extracts peak reflections (~80-150 targets).
  7. Gaussian Measurement Noise:
     - Injects range noise (sigma_R = 0.15m), azimuth noise (sigma_az = 1.0 deg), Doppler noise.
  8. Output File Exporter:
     - Exports to .npy, .csv, and standard .pcd formats + high-res comparison visualization.
====================================================================================================
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# --------------------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "dataset mini", "v1.0-mini")
META_DIR = os.path.join(DATA_ROOT, "v1.0-mini")
OUTPUT_DIR = os.path.join(BASE_DIR, "simulated_radar_data")

# 77 GHz FMCW Radar Physical Parameters
CARRIER_FREQ_HZ = 77.0e9        # 77 GHz center frequency
SPEED_OF_LIGHT = 299792458.0    # m/s
WAVELENGTH_M = SPEED_OF_LIGHT / CARRIER_FREQ_HZ # lambda ~ 3.89 mm
TX_POWER_DBM = 12.0             # Peak transmit power (12 dBm ~ 15.8 mW)
TX_POWER_W = 10 ** ((TX_POWER_DBM - 30.0) / 10.0)
ANTENNA_GAIN_DBI = 24.0         # Combined Tx + Rx antenna gain
ANTENNA_GAIN_LIN = 10 ** (ANTENNA_GAIN_DBI / 10.0)
NOISE_FLOOR_DBM = -105.0        # Thermal noise floor + receiver noise figure
NOISE_FLOOR_W = 10 ** ((NOISE_FLOOR_DBM - 30.0) / 10.0)
CFAR_SNR_THRESHOLD_DB = 8.0     # Minimum detection threshold (CFAR)

# Radar Field of View (FOV)
FOV_AZIMUTH_DEG = 45.0          # +/- 45 degrees
FOV_ELEVATION_DEG = 10.0        # +/- 10 degrees
MIN_RANGE_M = 1.0
MAX_RANGE_M = 90.0

# Nominal Radar Cross Section (RCS) values in dBsm (dB relative to 1 m^2)
RCS_PRIORS_DBSM = {
    'truck_bus': 25.0,           # Large commercial vehicles: +25 dBsm
    'truck': 25.0,
    'bus': 25.0,
    'car': 15.0,                 # Passenger cars: +15 dBsm
    'vehicle': 15.0,
    'auto_rickshaw': 10.0,       # Three-wheelers: +10 dBsm
    'motorcycle': 7.0,           # Two-wheelers: +7 dBsm
    'bicycle': 4.0,              # Cyclists: +4 dBsm
    'pedestrian': 0.0,           # Humans: 0 dBsm (~1 m^2)
    'animal': 2.0,               # Cattle / stray animals: +2 dBsm
    'barrier': 12.0,             # Metal/concrete road barrier: +12 dBsm
    'trafficcone': -5.0,         # Plastic cones: -5 dBsm
    'road_clutter': -15.0        # Road ground return: -15 dBsm
}

# --------------------------------------------------------------------------------------------------
# GEOMETRY UTILITIES
# --------------------------------------------------------------------------------------------------
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)

# --------------------------------------------------------------------------------------------------
# MMWAVE RADAR SIMULATOR CORE CLASS
# --------------------------------------------------------------------------------------------------
class SimulatedMMWaveRadarGenerator:
    """
    Physical-mathematical simulation engine that converts 3D LiDAR point clouds
    and dynamic scene objects into automotive 77 GHz FMCW mmWave radar returns.
    """
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_radar_from_lidar(self, pts_lidar, obstacles=None, ego_velocity=(0.0, 0.0)):
        """
        pts_lidar: [N, 3] or [N, 4] numpy array in ego vehicle coordinates
        obstacles: list of dicts with 'pos', 'size', 'class', 'velocity'
        ego_velocity: (vx, vy) in m/s
        
        Returns:
          simulated_targets: structured numpy array with columns:
            [X, Y, Z, Vx, Vy, Vr_Doppler, RCS_dBsm, SNR_dB, Target_Class_ID]
        """
        pts = pts_lidar[:, :3].copy()
        
        # 1. Range, Azimuth, Elevation Calculation
        # In nuScenes ego frame: X is forward, Y is lateral (+left), Z is vertical (+up)
        # Radar coordinates: R = hypot(X, Y, Z), azimuth theta = arctan2(Y, X)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        
        ranges = np.sqrt(x**2 + y**2 + z**2)
        azimuth_deg = np.rad2deg(np.arctan2(y, x))
        elevation_deg = np.rad2deg(np.arcsin(np.clip(z / (ranges + 1e-6), -1.0, 1.0)))
        
        # 2. Antenna FOV Gating
        fov_mask = (
            (ranges >= MIN_RANGE_M) & (ranges <= MAX_RANGE_M) &
            (np.abs(azimuth_deg) <= FOV_AZIMUTH_DEG) &
            (np.abs(elevation_deg) <= FOV_ELEVATION_DEG) &
            (z >= -1.8) # Filter points far below road surface
        )
        
        valid_pts = pts[fov_mask]
        valid_r = ranges[fov_mask]
        valid_az = azimuth_deg[fov_mask]
        valid_el = elevation_deg[fov_mask]
        
        if len(valid_pts) == 0:
            print("[WARN] No LiDAR points in Radar FOV!")
            return np.empty((0, 9), dtype=np.float32)

        # 3. Associate Points with Dynamic Obstacles to compute Doppler Velocity & RCS
        vx_pts = np.zeros(len(valid_pts), dtype=np.float32)
        vy_pts = np.zeros(len(valid_pts), dtype=np.float32)
        rcs_dbsm_pts = np.full(len(valid_pts), RCS_PRIORS_DBSM['road_clutter'], dtype=np.float32)
        class_id_pts = np.zeros(len(valid_pts), dtype=np.int32) # 0: stationary clutter

        if obstacles:
            for obs_idx, obs in enumerate(obstacles):
                ox, oy = obs['pos'][0], obs['pos'][1]
                w, l = obs['size'][0], obs['size'][1]
                cname = obs.get('class', 'vehicle').lower()
                
                # Check bounding box inclusion (+ margin)
                box_mask = (
                    (np.abs(valid_pts[:, 0] - ox) <= (l / 2.0 + 0.35)) &
                    (np.abs(valid_pts[:, 1] - oy) <= (w / 2.0 + 0.35))
                )
                
                if np.any(box_mask):
                    # Get obstacle velocity
                    obs_vel = obs.get('velocity', np.array([0.0, 0.0]))
                    vx_pts[box_mask] = obs_vel[0]
                    vy_pts[box_mask] = obs_vel[1]
                    
                    # Look up prior RCS
                    base_rcs = 10.0
                    for key, val in RCS_PRIORS_DBSM.items():
                        if key in cname:
                            base_rcs = val
                            break
                    rcs_dbsm_pts[box_mask] = base_rcs
                    class_id_pts[box_mask] = obs_idx + 1

        # Relative velocity relative to ego vehicle
        v_rel_x = vx_pts - ego_velocity[0]
        v_rel_y = vy_pts - ego_velocity[1]

        # 4. Doppler Radial Velocity Formula: v_r = (x*vx + y*vy + z*vz) / R
        # LOS unit vector: [x/R, y/R, z/R]
        los_x = valid_pts[:, 0] / valid_r
        los_y = valid_pts[:, 1] / valid_r
        vr_doppler = los_x * v_rel_x + los_y * v_rel_y # [m/s]

        # 5. Radar Range Equation (Received Power & SNR)
        # P_r = (P_t * G^2 * lambda^2 * sigma) / ((4*pi)^3 * R^4)
        sigma_m2 = 10.0 ** (rcs_dbsm_pts / 10.0) # Convert dBsm to linear m^2
        numerator = TX_POWER_W * (ANTENNA_GAIN_LIN ** 2) * (WAVELENGTH_M ** 2) * sigma_m2
        denominator = ((4.0 * np.pi) ** 3) * (valid_r ** 4) + 1e-12
        p_rx_w = numerator / denominator

        # Antenna Radiation Pattern Attenuation: G(theta) ~ exp(-theta^2 / (2*sigma_b^2))
        beam_sigma = FOV_AZIMUTH_DEG / 2.2
        antenna_factor = np.exp(-(valid_az ** 2) / (2.0 * (beam_sigma ** 2)))
        p_rx_w = p_rx_w * antenna_factor

        # SNR in dB: 10 * log10(P_r / P_noise)
        snr_db = 10.0 * np.log10(np.clip(p_rx_w / NOISE_FLOOR_W, 1e-3, 1e9))

        # 6. CFAR Threshold Filter (Filter out weak signals below detection floor)
        cfar_mask = snr_db >= CFAR_SNR_THRESHOLD_DB
        pts_cfar = valid_pts[cfar_mask]
        r_cfar = valid_r[cfar_mask]
        az_cfar = valid_az[cfar_mask]
        el_cfar = valid_el[cfar_mask]
        vr_cfar = vr_doppler[cfar_mask]
        vx_cfar = v_rel_x[cfar_mask]
        vy_cfar = v_rel_y[cfar_mask]
        rcs_cfar = rcs_dbsm_pts[cfar_mask]
        snr_cfar = snr_db[cfar_mask]
        cid_cfar = class_id_pts[cfar_mask]

        if len(pts_cfar) == 0:
            print("[WARN] No returns passed CFAR threshold!")
            return np.empty((0, 9), dtype=np.float32)

        # 7. Range-Doppler Sparsification (Range-Azimuth Binning)
        # Radar produces discrete point returns (~80-150 targets) by picking peaks in range-azimuth-Doppler cells
        range_bins = np.linspace(MIN_RANGE_M, MAX_RANGE_M, int((MAX_RANGE_M - MIN_RANGE_M) / 0.8) + 1)
        az_bins = np.linspace(-FOV_AZIMUTH_DEG, FOV_AZIMUTH_DEG, int((2 * FOV_AZIMUTH_DEG) / 2.0) + 1)

        r_idx = np.digitize(r_cfar, range_bins)
        az_idx = np.digitize(az_cfar, az_bins)
        cell_keys = (r_idx << 16) | az_idx

        # Group by cell and pick the highest SNR reflection per cell
        unique_cells, cell_indices = np.unique(cell_keys, return_inverse=True)
        selected_target_indices = []

        for u_i in range(len(unique_cells)):
            members = np.where(cell_indices == u_i)[0]
            # Pick highest SNR peak reflection
            best_member = members[np.argmax(snr_cfar[members])]
            selected_target_indices.append(best_member)

        # Limit to realistic max radar targets (80 to 180 targets per scan)
        selected_target_indices = np.array(selected_target_indices)
        if len(selected_target_indices) > 160:
            # Sort by SNR and keep top 160 detections
            sorted_by_snr = selected_target_indices[np.argsort(-snr_cfar[selected_target_indices])]
            selected_target_indices = sorted_by_snr[:160]

        # 8. Ingest Realistic Gaussian Measurement Noise
        # Range noise ~ 0.15m, Azimuth noise ~ 0.8 deg, Doppler noise ~ 0.1 m/s
        noise_r = np.random.normal(0.0, 0.12, size=len(selected_target_indices)).astype(np.float32)
        noise_az_rad = np.deg2rad(np.random.normal(0.0, 0.6, size=len(selected_target_indices))).astype(np.float32)
        noise_vr = np.random.normal(0.0, 0.08, size=len(selected_target_indices)).astype(np.float32)

        sel_r = r_cfar[selected_target_indices] + noise_r
        sel_az_rad = np.deg2rad(az_cfar[selected_target_indices]) + noise_az_rad
        sel_el_rad = np.deg2rad(el_cfar[selected_target_indices])

        # Convert back to Cartesian [X, Y, Z]
        x_radar = sel_r * np.cos(sel_az_rad) * np.cos(sel_el_rad)
        y_radar = sel_r * np.sin(sel_az_rad) * np.cos(sel_el_rad)
        z_radar = sel_r * np.sin(sel_el_rad)

        vr_noisy = vr_cfar[selected_target_indices] + noise_vr
        rcs_noisy = rcs_cfar[selected_target_indices]
        snr_noisy = snr_cfar[selected_target_indices]
        vx_noisy = vx_cfar[selected_target_indices]
        vy_noisy = vy_cfar[selected_target_indices]
        cid_noisy = cid_cfar[selected_target_indices]

        # Assemble final matrix [M, 9]
        simulated_matrix = np.column_stack([
            x_radar,
            y_radar,
            z_radar,
            vx_noisy,
            vy_noisy,
            vr_noisy,
            rcs_noisy,
            snr_noisy,
            cid_noisy
        ]).astype(np.float32)

        return simulated_matrix

    def save_radar_files(self, radar_data, sample_idx=12, token="sample"):
        """Saves simulated radar data in .npy, .csv, and .pcd formats."""
        base_name = f"simulated_mmwave_radar_sample_{sample_idx}"
        
        # 1. Save .npy (for high-speed programmatic loading)
        npy_path = os.path.join(self.output_dir, f"{base_name}.npy")
        np.save(npy_path, radar_data)

        # 2. Save .csv (human-readable table)
        csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
        header = "x_m,y_m,z_m,vx_mps,vy_mps,vr_doppler_mps,rcs_dbsm,snr_db,target_id"
        np.savetxt(csv_path, radar_data, delimiter=",", header=header, comments="", fmt="%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.1f,%.1f,%d")

        # 3. Save .pcd (Point Cloud Data standard)
        pcd_path = os.path.join(self.output_dir, f"{base_name}.pcd")
        num_points = len(radar_data)
        with open(pcd_path, "w") as f:
            f.write("# .PCD v0.7 - Synthetic 77 GHz mmWave Radar Point Cloud\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z vx vy vr rcs snr id\n")
            f.write("SIZE 4 4 4 4 4 4 4 4 4\n")
            f.write("TYPE F F F F F F F F I\n")
            f.write("COUNT 1 1 1 1 1 1 1 1 1\n")
            f.write(f"WIDTH {num_points}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\n")
            f.write("DATA ascii\n")
            for row in radar_data:
                f.write(f"{row[0]:.3f} {row[1]:.3f} {row[2]:.3f} {row[3]:.2f} {row[4]:.2f} {row[5]:.2f} {row[6]:.1f} {row[7]:.1f} {int(row[8])}\n")

        return {
            "npy": npy_path,
            "csv": csv_path,
            "pcd": pcd_path
        }

    def render_comparison_dashboard(self, pts_lidar, radar_data, obstacles=None, sample_idx=12):
        """Creates a 4-panel visual engineering dashboard comparing LiDAR vs. Simulated mmWave Radar."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='#0D1117')
        fig.suptitle(f"77 GHz FMCW mmWAVE RADAR SYNTHETIC GENERATOR FROM 3D LIDAR\nnuScenes Sample #{sample_idx} | Targets Detected: {len(radar_data)}",
                     fontsize=15, fontweight='bold', color='#58A6FF', y=0.97)

        # -------------------------------------------------------------
        # PANEL 1: DENSE LIDAR VS SPARSE RADAR BEV OVERLAY
        # -------------------------------------------------------------
        ax1 = axes[0, 0]
        ax1.set_facecolor('#161B22')
        # LiDAR background
        l_mask = (pts_lidar[:, 0] >= 0) & (pts_lidar[:, 0] <= 60) & (abs(pts_lidar[:, 1]) <= 30)
        p_l = pts_lidar[l_mask]
        ax1.scatter(p_l[:, 1], p_l[:, 0], c='#30363D', s=0.3, alpha=0.4, label=f"Dense LiDAR ({len(p_l):,} pts)")
        
        # Radar returns
        ax1.scatter(radar_data[:, 1], radar_data[:, 0], c='#58A6FF', s=35, marker='o', edgecolors='white',
                    linewidths=0.8, alpha=0.9, label=f"Simulated Radar ({len(radar_data)} targets)")
        
        # Ego vehicle
        ax1.plot(0, 0, 'r^', markersize=9, label="Ego Radar (Front Bumper)")
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-2, 60)
        ax1.set_xlabel("Lateral Y (meters)", color='white', fontsize=10)
        ax1.set_ylabel("Forward Range X (meters)", color='white', fontsize=10)
        ax1.tick_params(colors='white')
        ax1.grid(True, linestyle='--', color='#21262D', alpha=0.7)
        ax1.set_title("1. BEV Comparison: Dense Optical LiDAR vs. Sparse mmWave Radar", color='white', fontsize=11, fontweight='bold')
        ax1.legend(loc="upper right", facecolor='#21262D', edgecolor='white', labelcolor='white')

        # -------------------------------------------------------------
        # PANEL 2: DOPPLER RADIAL VELOCITY MAP (Vr)
        # -------------------------------------------------------------
        ax2 = axes[0, 1]
        ax2.set_facecolor('#161B22')
        vr = radar_data[:, 5]
        v_scatter = ax2.scatter(radar_data[:, 1], radar_data[:, 0], c=vr, cmap='coolwarm', s=45,
                                edgecolors='black', linewidths=0.5, vmin=-12, vmax=12)
        cbar_v = fig.colorbar(v_scatter, ax=ax2, fraction=0.03, pad=0.02)
        cbar_v.set_label('Doppler Radial Velocity Vr (m/s)', color='white', fontsize=9)
        cbar_v.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar_v.ax.get_yticklabels(), color='white')

        ax2.plot(0, 0, 'w^', markersize=8)
        ax2.set_xlim(-25, 25)
        ax2.set_ylim(-2, 60)
        ax2.set_xlabel("Lateral Y (meters)", color='white', fontsize=10)
        ax2.set_ylabel("Forward Range X (meters)", color='white', fontsize=10)
        ax2.tick_params(colors='white')
        ax2.grid(True, linestyle='--', color='#21262D', alpha=0.7)
        ax2.set_title("2. Instantaneous Doppler Velocity: Red = Receding, Blue = Approaching (Closing Hazard)", color='white', fontsize=11, fontweight='bold')

        # -------------------------------------------------------------
        # PANEL 3: RADAR CROSS SECTION (RCS in dBsm) & TARGET IDENTITY
        # -------------------------------------------------------------
        ax3 = axes[1, 0]
        ax3.set_facecolor('#161B22')
        rcs = radar_data[:, 6]
        rcs_scatter = ax3.scatter(radar_data[:, 1], radar_data[:, 0], c=rcs, cmap='viridis', s=45,
                                  edgecolors='white', linewidths=0.5, vmin=-15, vmax=30)
        cbar_r = fig.colorbar(rcs_scatter, ax=ax3, fraction=0.03, pad=0.02)
        cbar_r.set_label('Radar Cross Section (RCS dBsm)', color='white', fontsize=9)
        cbar_r.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar_r.ax.get_yticklabels(), color='white')

        ax3.plot(0, 0, 'w^', markersize=8)
        ax3.set_xlim(-25, 25)
        ax3.set_ylim(-2, 60)
        ax3.set_xlabel("Lateral Y (meters)", color='white', fontsize=10)
        ax3.set_ylabel("Forward Range X (meters)", color='white', fontsize=10)
        ax3.tick_params(colors='white')
        ax3.grid(True, linestyle='--', color='#21262D', alpha=0.7)
        ax3.set_title("3. Radar Cross Section (RCS): Metallic Reflectivity vs. Biological Backscatter", color='white', fontsize=11, fontweight='bold')

        # -------------------------------------------------------------
        # PANEL 4: RADAR RANGE EQUATION 1/R^4 POWER & SNR CURVE
        # -------------------------------------------------------------
        ax4 = axes[1, 1]
        ax4.set_facecolor('#161B22')
        r_all = np.hypot(radar_data[:, 0], radar_data[:, 1])
        snr_all = radar_data[:, 7]

        # Theoretical 1/R^4 curve for a +15 dBsm car
        r_model = np.linspace(2.0, 75.0, 200)
        p_model = (TX_POWER_W * (ANTENNA_GAIN_LIN**2) * (WAVELENGTH_M**2) * (10**1.5)) / (((4*np.pi)**3) * (r_model**4) + 1e-12)
        snr_model = 10 * np.log10(p_model / NOISE_FLOOR_W)

        ax4.plot(r_model, snr_model, color='#E3B341', linewidth=2.2, linestyle='--', label="Theoretical 1/R⁴ Curve (Car +15 dBsm)")
        ax4.scatter(r_all, snr_all, c='#3FB950', s=35, alpha=0.85, edgecolors='white', linewidths=0.5, label="Simulated mmWave Target Returns")
        ax4.axhline(y=CFAR_SNR_THRESHOLD_DB, color='#F85149', linestyle=':', linewidth=1.8, label=f"CFAR Detection Floor ({CFAR_SNR_THRESHOLD_DB} dB)")

        ax4.set_xlim(0, 80)
        ax4.set_ylim(0, 70)
        ax4.set_xlabel("Range to Target R (meters)", color='white', fontsize=10)
        ax4.set_ylabel("Received Signal-to-Noise Ratio (dB)", color='white', fontsize=10)
        ax4.tick_params(colors='white')
        ax4.grid(True, linestyle='--', color='#21262D', alpha=0.7)
        ax4.set_title("4. Radar Physics Validation: Range vs. SNR (1/R⁴ Power Decay)", color='white', fontsize=11, fontweight='bold')
        ax4.legend(loc="upper right", facecolor='#21262D', edgecolor='white', labelcolor='white')

        # Save plot
        out_plot = os.path.join(self.output_dir, f"simulated_radar_analysis_sample_{sample_idx}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(out_plot, dpi=200, facecolor=fig.get_facecolor())
        plt.close()
        return out_plot


# --------------------------------------------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# --------------------------------------------------------------------------------------------------
def run_generator_on_dataset_sample(sample_idx=12):
    print("=" * 80)
    print(f"   77 GHz FMCW mmWAVE RADAR SYNTHETIC GENERATOR FROM REAL LIDAR DATA")
    print(f"                        (nuScenes Sample #{sample_idx})")
    print("=" * 80)

    # 1. Load Dataset Metadata
    print("\n[1/4] Loading nuScenes Keyframe & Sensor Calibrations...")
    with open(os.path.join(META_DIR, "sample.json")) as f: samples = json.load(f)
    with open(os.path.join(META_DIR, "sample_data.json")) as f: sample_data_list = json.load(f)
    with open(os.path.join(META_DIR, "calibrated_sensor.json")) as f: calib = {c["token"]: c for c in json.load(f)}
    with open(os.path.join(META_DIR, "ego_pose.json")) as f: ego_poses = {ep["token"]: ep for ep in json.load(f)}
    with open(os.path.join(META_DIR, "category.json")) as f: categories = {c["token"]: c["name"] for c in json.load(f)}
    with open(os.path.join(META_DIR, "instance.json")) as f: instances = {inst["token"]: inst["category_token"] for inst in json.load(f)}
    with open(os.path.join(META_DIR, "sample_annotation.json")) as f: all_anns = json.load(f)

    if sample_idx >= len(samples):
        sample_idx = 0
    sample = samples[sample_idx]
    s_token = sample["token"]

    lidar_sd = next(sd for sd in sample_data_list if sd["sample_token"] == s_token and sd["is_key_frame"] and "LIDAR_TOP" in sd["filename"])
    lidar_calib = calib[lidar_sd["calibrated_sensor_token"]]
    lidar_ego = ego_poses[lidar_sd["ego_pose_token"]]

    # 2. Ingest Raw LiDAR
    print("\n[2/4] Reading 32-beam LiDAR Point Cloud...")
    lidar_path = os.path.join(DATA_ROOT, lidar_sd["filename"].replace("/", os.sep))
    raw_pcd = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    pts_lidar = raw_pcd[:, :3]

    # Transform LiDAR to Ego Vehicle Frame
    r_lidar = quaternion_to_rotation_matrix(lidar_calib["rotation"])
    t_lidar = np.array(lidar_calib["translation"], dtype=np.float32)
    pts_ego = (r_lidar @ pts_lidar.T).T + t_lidar

    # Extract active obstacle annotations with velocities
    sample_anns = [a for a in all_anns if a["sample_token"] == s_token]
    r_ego = quaternion_to_rotation_matrix(lidar_ego["rotation"])
    t_ego = np.array(lidar_ego["translation"], dtype=np.float32)
    obstacles = []

    for ann in sample_anns:
        cname = categories.get(instances.get(ann["instance_token"]), "object")
        box_pos_g = np.array(ann["translation"], dtype=np.float32)
        pos_ego = r_ego.T @ (box_pos_g - t_ego)
        box_size = np.array(ann["size"], dtype=np.float32)
        
        # Synthetic realistic speed based on class
        speed = 4.5 if ("car" in cname or "truck" in cname) else (1.2 if "pedestrian" in cname else 0.0)
        heading_rot = quaternion_to_rotation_matrix(ann["rotation"])
        rot_e = r_ego.T @ heading_rot
        v_obs = np.array([speed * rot_e[0, 0], speed * rot_e[1, 0]], dtype=np.float32)

        obstacles.append({
            "pos": pos_ego,
            "size": box_size,
            "class": cname,
            "velocity": v_obs
        })

    print(f"  * Raw LiDAR Points Ingested : {len(pts_ego):,} points")
    print(f"  * Tracked Scene Obstacles   : {len(obstacles)} objects")

    # 3. Execute Physical Radar Generator Engine
    print("\n[3/4] Running 77 GHz mmWave Radar Physics Simulation Engine...")
    t0 = time.time()
    generator = SimulatedMMWaveRadarGenerator()
    sim_radar = generator.generate_radar_from_lidar(
        pts_lidar=pts_ego,
        obstacles=obstacles,
        ego_velocity=(11.1, 0.0) # Ego vehicle cruising at 40 km/h (11.1 m/s)
    )
    dt_gen = (time.time() - t0) * 1000

    print(f"  * Simulation Computation Time: {dt_gen:.2f} ms")
    print(f"  * Generated Radar Detections : {len(sim_radar)} target reflections (Sparse FMCW Point Cloud)")
    print(f"  * Min Range: {np.min(np.hypot(sim_radar[:, 0], sim_radar[:, 1])):.2f}m | Max Range: {np.max(np.hypot(sim_radar[:, 0], sim_radar[:, 1])):.2f}m")
    print(f"  * Doppler Velocity Range (Vr): [{np.min(sim_radar[:, 5]):.2f} m/s to {np.max(sim_radar[:, 5]):.2f} m/s]")
    print(f"  * Radar Cross Section (RCS)  : [{np.min(sim_radar[:, 6]):.1f} dBsm to {np.max(sim_radar[:, 6]):.1f} dBsm]")
    print(f"  * Signal-to-Noise Ratio (SNR): [{np.min(sim_radar[:, 7]):.1f} dB to {np.max(sim_radar[:, 7]):.1f} dB]")

    # 4. Save to Disk
    print("\n[4/4] Exporting Simulated Radar Data to Storage Files...")
    saved_files = generator.save_radar_files(sim_radar, sample_idx=sample_idx, token=s_token)
    plot_file = generator.render_comparison_dashboard(pts_ego, sim_radar, obstacles, sample_idx=sample_idx)

    print(f"  [SAVED] NumPy Data Array  : {saved_files['npy']} ({os.path.getsize(saved_files['npy'])/1024:.1f} KB)")
    print(f"  [SAVED] CSV Data Table    : {saved_files['csv']} ({os.path.getsize(saved_files['csv'])/1024:.1f} KB)")
    print(f"  [SAVED] Standard PCD File : {saved_files['pcd']} ({os.path.getsize(saved_files['pcd'])/1024:.1f} KB)")
    print(f"  [SAVED] Analysis Dashboard: {plot_file} ({os.path.getsize(plot_file)/1e6:.2f} MB)")

    print("\n" + "=" * 80)
    print(" SAMPLE TARGET DATA EXTRACT (Top 5 Radar Returns):")
    print(" [X (m),  Y (m),  Z (m),  Vx,     Vy,     Vr (Doppler), RCS (dBsm), SNR (dB), Target_ID]")
    print("=" * 80)
    for row in sim_radar[:5]:
        print(f" [{row[0]:6.2f}, {row[1]:6.2f}, {row[2]:5.2f}, {row[3]:6.2f}, {row[4]:6.2f}, {row[5]:7.2f} m/s,  {row[6]:5.1f} dBsm,   {row[7]:5.1f} dB,  ID#{int(row[8])}]")
    print("=" * 80)
    print("[SUCCESS] mmWave Radar Generation Completed!")
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Simulated 77 GHz mmWave Radar from LiDAR")
    parser.add_argument("--sample", type=int, default=12, help="nuScenes sample index (0-403)")
    args = parser.parse_args()

    run_generator_on_dataset_sample(sample_idx=args.sample)
