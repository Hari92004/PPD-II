"""
Script to generate high-resolution (300 DPI) scientific diagram for:
Perception, Tracking & Predictive Intelligence
(Spatial Calibration -> EKF Tracking -> 4-Expert MoE Prediction -> Collision Risk)
Matches the slide 'Perception, Tracking & Predictive Intelligence'
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import shutil
import os

def create_perception_tracking_diagram(output_path=r"D:\PPD2\perception_tracking_prediction_architecture.png"):
    fig = plt.figure(figsize=(19, 10.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # Main Diagram Header
    fig.text(0.5, 0.965, "PERCEPTION, TRACKING & PREDICTIVE INTELLIGENCE",
             ha='center', va='center', fontsize=20, fontweight='bold', color='#0f172a', fontname='DejaVu Sans')
    fig.text(0.5, 0.935, "Layer 2 to Layer 3 Architecture: Multimodal Fusion, CTRA EKF Tracking, 4-Expert MoE & TTC Risk Assessment",
             ha='center', va='center', fontsize=11.5, color='#475569', fontname='DejaVu Sans')

    # Outer Layout: 4 Columns representing the 4 stages on the slide
    # Column 1: Spatial Calibration & Multimodal Fusion (w: 0.22)
    # Column 2: Extended Kalman Filter (EKF) Tracking (w: 0.22)
    # Column 3: 4-Expert MoE Motion Prediction (w: 0.30)
    # Column 4: Collision Risk Assessment (w: 0.18)
    
    col_w = [0.21, 0.22, 0.31, 0.19]
    col_gap = 0.02
    left_margin = 0.03
    top_pos = 0.89
    bottom_pos = 0.06
    card_h = top_pos - bottom_pos

    col_x = [
        left_margin,
        left_margin + col_w[0] + col_gap,
        left_margin + col_w[0] + col_gap + col_w[1] + col_gap,
        left_margin + col_w[0] + col_gap + col_w[1] + col_gap + col_w[2] + col_gap
    ]

    # Helper function to create styled card background
    def add_card(ax_bg, x, y, w, h, title, subtitle, step_num):
        # Card shadow & border
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                      facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1.5,
                                      transform=fig.transFigure, clip_on=False, zorder=1)
        fig.patches.append(rect)
        
        # Header strip
        strip = patches.FancyBboxPatch((x, y + h - 0.08), w, 0.08, boxstyle="round,pad=0.01,rounding_size=0.012",
                                       facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.0,
                                       transform=fig.transFigure, clip_on=False, zorder=2)
        fig.patches.append(strip)
        
        # Step Number Badge
        badge = patches.Circle((x + 0.025, y + h - 0.04), 0.018, facecolor='#0f172a', edgecolor='none',
                               transform=fig.transFigure, clip_on=False, zorder=3)
        fig.patches.append(badge)
        fig.text(x + 0.025, y + h - 0.04, str(step_num), ha='center', va='center',
                 fontsize=12, fontweight='bold', color='#ffffff', zorder=4)

        # Title & Subtitle
        fig.text(x + 0.05, y + h - 0.03, title, fontsize=12, fontweight='bold', color='#0f172a', zorder=4)
        fig.text(x + 0.05, y + h - 0.058, subtitle, fontsize=8.5, color='#64748b', zorder=4)

    # Add 4 Column Cards
    add_card(fig, col_x[0], bottom_pos, col_w[0], card_h, "1. Spatial Calibration", "Multimodal Unified Fusion", 1)
    add_card(fig, col_x[1], bottom_pos, col_w[1], card_h, "2. EKF State Tracking", "CTRA Kinematic Filtering", 2)
    add_card(fig, col_x[2], bottom_pos, col_w[2], card_h, "3. 4-Expert MoE Prediction", "Class-Specific Kinematic Nets", 3)
    add_card(fig, col_x[3], bottom_pos, col_w[3], card_h, "4. Collision Risk Engine", "2.4s Trajectory Horizon TTC", 4)

    # =========================================================================
    # STAGE 1: SPATIAL CALIBRATION & MULTIMODAL FUSION
    # =========================================================================
    ax1 = fig.add_axes([col_x[0] + 0.01, bottom_pos + 0.02, col_w[0] - 0.02, card_h - 0.11])
    ax1.set_axis_off()
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # 3 Sensor Blocks
    sensors = [
        ("Front Optical Camera", "1080p @ 30 FPS\nRGB Optical Semantics", "#f1f5f9"),
        ("Solid-State 3D LiDAR", "Metric Range Point Cloud\n300k pts/sec (.npy)", "#f1f5f9"),
        ("77 GHz FMCW Radar", "Doppler Range-Rate (vr)\nVirtual Physics Module", "#f1f5f9")
    ]
    for idx, (s_title, s_desc, bg_col) in enumerate(sensors):
        sy = 7.6 - idx * 2.1
        box = patches.FancyBboxPatch((0.5, sy), 9.0, 1.6, boxstyle="round,pad=0.2",
                                     facecolor=bg_col, edgecolor='#94a3b8', linewidth=1.2)
        ax1.add_patch(box)
        ax1.text(0.9, sy + 1.15, s_title, fontsize=9.5, fontweight='bold', color='#0f172a')
        ax1.text(0.9, sy + 0.45, s_desc, fontsize=8.0, color='#475569', linespacing=1.2)

    # Transformation Matrix Box
    tf_box = patches.FancyBboxPatch((0.5, 0.6), 9.0, 2.3, boxstyle="round,pad=0.2",
                                    facecolor='#0f172a', edgecolor='none')
    ax1.add_patch(tf_box)
    ax1.text(5.0, 2.35, "Rigid Transformation Matrix", ha='center', fontsize=9.5, fontweight='bold', color='#f8fafc')
    ax1.text(5.0, 1.65, r"$P_{proj} = K \cdot [ R \mid T ] \cdot P_{lidar}$", ha='center', fontsize=10.5, color='#38bdf8', fontweight='bold')
    ax1.text(5.0, 1.0, "Empirical Reprojection Error: 1.42 px\nUnified 3D BEV Metric Coordinate Frame",
             ha='center', fontsize=7.8, color='#cbd5e1', linespacing=1.2)

    # Arrow to Column 2
    fig.text(col_x[0] + col_w[0] + col_gap * 0.25, 0.50, r"$\mathbf{\rightarrow}$", fontsize=24, color='#94a3b8', fontweight='bold')

    # =========================================================================
    # STAGE 2: EXTENDED KALMAN FILTER (EKF) TRACKING
    # =========================================================================
    ax2 = fig.add_axes([col_x[1] + 0.01, bottom_pos + 0.02, col_w[1] - 0.02, card_h - 0.11])
    ax2.set_axis_off()
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # State Vector Panel
    sv_box = patches.FancyBboxPatch((0.5, 7.3), 9.0, 2.2, boxstyle="round,pad=0.2",
                                    facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.2)
    ax2.add_patch(sv_box)
    ax2.text(0.8, 8.95, "6D CTRA State Vector:", fontsize=9.5, fontweight='bold', color='#0f172a')
    ax2.text(5.0, 8.1, r"$\mathbf{x}_k = [ p_x,\; p_y,\; v,\; \psi,\; \dot{\psi},\; a ]^T$",
             ha='center', fontsize=11, fontweight='bold', color='#0f172a')
    ax2.text(5.0, 7.45, "Position (px, py) | Speed (v) | Yaw (ψ, ψ̇) | Accel (a)",
             ha='center', fontsize=7.6, color='#64748b')

    # 2-Stage Filter Cycle Diagram
    cycle_box = patches.FancyBboxPatch((0.5, 3.4), 9.0, 3.5, boxstyle="round,pad=0.2",
                                       facecolor='#ffffff', edgecolor='#94a3b8', linewidth=1.2)
    ax2.add_patch(cycle_box)
    ax2.text(5.0, 6.45, "Recursive 2-Stage Filter Cycle", ha='center', fontsize=9.5, fontweight='bold', color='#0f172a')

    # Prediction Sub-box
    p_box = patches.Rectangle((1.0, 4.9), 8.0, 1.25, facecolor='#f1f5f9', edgecolor='#cbd5e1')
    ax2.add_patch(p_box)
    ax2.text(1.3, 5.75, "1. Time Prediction:", fontsize=8.5, fontweight='bold', color='#0f172a')
    ax2.text(1.3, 5.15, r"$\mathbf{x}_{k|k-1} = f(\mathbf{x}_{k-1}), \;\; P_{k|k-1} = F P F^T + Q$", fontsize=8.2, color='#334155')

    # Update Sub-box
    u_box = patches.Rectangle((1.0, 3.5), 8.0, 1.25, facecolor='#f1f5f9', edgecolor='#cbd5e1')
    ax2.add_patch(u_box)
    ax2.text(1.3, 4.35, "2. Measurement Update:", fontsize=8.5, fontweight='bold', color='#0f172a')
    ax2.text(1.3, 3.75, r"$\mathbf{x}_{k|k} = \mathbf{x} + K_k(z_k - h(\mathbf{x})), \;\; K_k = P H^T S^{-1}$", fontsize=8.2, color='#334155')

    # Benefit Highlights
    ben_box = patches.FancyBboxPatch((0.5, 0.6), 9.0, 2.4, boxstyle="round,pad=0.2",
                                     facecolor='#0f172a', edgecolor='none')
    ax2.add_patch(ben_box)
    ax2.text(5.0, 2.4, "Operational Benefits", ha='center', fontsize=9.5, fontweight='bold', color='#f8fafc')
    ax2.text(0.9, 1.75, "• Ultra-Low CPU Latency: 0.35 ms / object\n• Eliminates frame-to-frame detector jitter\n• Maintains tracks during temporary occlusion\n• Deterministic ISO 26262 functional safety",
             fontsize=7.8, color='#e2e8f0', linespacing=1.35)

    # Arrow to Column 3
    fig.text(col_x[1] + col_w[1] + col_gap * 0.25, 0.50, r"$\mathbf{\rightarrow}$", fontsize=24, color='#94a3b8', fontweight='bold')

    # =========================================================================
    # STAGE 3: 4-EXPERT MoE MOTION PREDICTION
    # =========================================================================
    ax3 = fig.add_axes([col_x[2] + 0.01, bottom_pos + 0.02, col_w[2] - 0.02, card_h - 0.11])
    ax3.set_axis_off()
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    # Gating & Classification Header
    gate_box = patches.FancyBboxPatch((0.5, 8.4), 9.0, 1.15, boxstyle="round,pad=0.15",
                                      facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.2)
    ax3.add_patch(gate_box)
    ax3.text(5.0, 9.15, "Class Gating Router: Routes Track Vector by Traffic Class", ha='center', fontsize=9.0, fontweight='bold', color='#0f172a')
    ax3.text(5.0, 8.65, "Input: [ px, py, v, ψ, a ] across 10-frame sliding window (0.5s history)", ha='center', fontsize=7.6, color='#64748b')

    # 4 Expert Cards
    experts = [
        ("EXPERT 1: Pedestrians & Cattle", "Architecture: Gated Recurrent Unit (GRU)",
         "Captures high-entropy erratic motion, sudden roadside halts, and rapid direction reversals.",
         "ADE: 0.42 m | Latency: 4.8 ms", "#ffffff"),
        
        ("EXPERT 2: Two-Wheelers & Auto-Rickshaws", "Architecture: Spatio-Temporal GCN (ST-GCN)",
         "Models spatial lane splitting, chaotic gap intrusions, and rapid slalom maneuvers.",
         "ADE: 0.58 m | Latency: 6.2 ms", "#ffffff"),
        
        ("EXPERT 3: Cars & Passenger Vehicles", "Architecture: Physics-Informed Neural Net (PINN)",
         "Enforces Ackermann kinematic steering constraints and tire slip angle limits.",
         "ADE: 0.36 m | Latency: 5.1 ms", "#ffffff"),
        
        ("EXPERT 4: Buses & Heavy Commercial Trucks", "Architecture: Multi-Head LSTM + Swept-Path",
         "Accounts for heavy inertia, delayed deceleration, and large rear-wheel swept envelope.",
         "ADE: 0.64 m | Latency: 6.3 ms", "#ffffff")
    ]

    for idx, (e_title, e_arch, e_desc, e_met, bg_c) in enumerate(experts):
        ey = 6.3 - idx * 1.95
        eb = patches.FancyBboxPatch((0.5, ey), 9.0, 1.75, boxstyle="round,pad=0.18",
                                    facecolor=bg_c, edgecolor='#94a3b8', linewidth=1.1)
        ax3.add_patch(eb)
        
        # Expert title badge
        ax3.text(0.9, ey + 1.35, e_title, fontsize=8.8, fontweight='bold', color='#0f172a')
        ax3.text(0.9, ey + 0.95, e_arch, fontsize=7.8, fontweight='bold', color='#0284c7')
        ax3.text(0.9, ey + 0.55, e_desc, fontsize=7.2, color='#475569')
        ax3.text(0.9, ey + 0.15, e_met, fontsize=7.2, fontweight='bold', color='#16a34a')

    # MoE Aggregation Footer
    agg_box = patches.FancyBboxPatch((0.5, 0.45), 9.0, 1.25, boxstyle="round,pad=0.15",
                                     facecolor='#0f172a', edgecolor='none')
    ax3.add_patch(agg_box)
    ax3.text(5.0, 1.2, "2.4-Second Trajectory Horizon Rollout", ha='center', fontsize=9.2, fontweight='bold', color='#f8fafc')
    ax3.text(5.0, 0.68, "Generates 12 future waypoints (5 Hz) with 2D Gaussian uncertainty covariance",
             ha='center', fontsize=7.6, color='#cbd5e1')

    # Arrow to Column 4
    fig.text(col_x[2] + col_w[2] + col_gap * 0.25, 0.50, r"$\mathbf{\rightarrow}$", fontsize=24, color='#94a3b8', fontweight='bold')

    # =========================================================================
    # STAGE 4: COLLISION RISK ASSESSMENT (TTC)
    # =========================================================================
    ax4 = fig.add_axes([col_x[3] + 0.01, bottom_pos + 0.02, col_w[3] - 0.02, card_h - 0.11])
    ax4.set_axis_off()
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    # TTC Formula Box
    ttc_box = patches.FancyBboxPatch((0.5, 7.3), 9.0, 2.2, boxstyle="round,pad=0.2",
                                     facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.2)
    ax4.add_patch(ttc_box)
    ax4.text(5.0, 8.95, "Time-to-Collision (TTC)", ha='center', fontsize=9.5, fontweight='bold', color='#0f172a')
    ax4.text(5.0, 8.1, r"$\mathrm{TTC} = \frac{d_{rel}}{v_{rel}} = \frac{\| \mathbf{p}_{ego} - \mathbf{p}_{obj} \|_2}{v_{ego} - v_{obj}}$",
             ha='center', fontsize=10.5, fontweight='bold', color='#0f172a')
    ax4.text(5.0, 7.45, "Evaluated across all 12 horizon waypoints", ha='center', fontsize=7.5, color='#64748b')

    # 3 Safety Tiers
    tiers = [
        ("Green Tier: SAFE", "TTC > 3.0 s", "Normal cruise along lane center", "#f0fdf4", "#16a34a"),
        ("Amber Tier: CAUTION", "2.0 s < TTC ≤ 3.0 s", "Prepare avoidance; advisory HUD alert", "#fffbeb", "#d97706"),
        ("Red Tier: CRITICAL", "TTC ≤ 2.0 s or d < dcrit", "Triggers 3-Zone Quintic Evasion", "#fef2f2", "#dc2626")
    ]
    for idx, (t_name, t_cond, t_act, t_bg, t_border) in enumerate(tiers):
        ty = 5.2 - idx * 1.55
        tb = patches.FancyBboxPatch((0.5, ty), 9.0, 1.35, boxstyle="round,pad=0.15",
                                    facecolor=t_bg, edgecolor=t_border, linewidth=1.2)
        ax4.add_patch(tb)
        ax4.text(0.8, ty + 0.95, t_name, fontsize=8.5, fontweight='bold', color=t_border)
        ax4.text(0.8, ty + 0.55, f"Condition: {t_cond}", fontsize=7.5, fontweight='bold', color='#1e293b')
        ax4.text(0.8, ty + 0.18, t_act, fontsize=7.0, color='#475569')

    # Hand-off to Layer 4 Box
    handoff_box = patches.FancyBboxPatch((0.5, 0.45), 9.0, 2.0, boxstyle="round,pad=0.2",
                                         facecolor='#0f172a', edgecolor='none')
    ax4.add_patch(handoff_box)
    ax4.text(5.0, 1.95, "Layer 4 Hand-off", ha='center', fontsize=9.5, fontweight='bold', color='#f8fafc')
    ax4.text(5.0, 1.4, "Sends Collision Point (s_coll, d_coll)\nto 3-Zone Frenet Path Planner",
             ha='center', fontsize=7.8, color='#38bdf8', fontweight='bold')
    ax4.text(5.0, 0.75, "Ego initiates quintic evasion curve:\nq(t) = a0 + a1*t + ... + a5*t^5",
             ha='center', fontsize=7.3, color='#cbd5e1')

    # Save high-res output
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Perception & Tracking Architecture diagram successfully generated at: {output_path}")

    # Copy to Brain Artifacts folder for preview
    brain_dir = r"C:\Users\harip\.gemini\antigravity\brain\2d6e9244-0990-4251-8a0a-dd1256a14492"
    if os.path.exists(brain_dir):
        dest_brain = os.path.join(brain_dir, "perception_tracking_prediction_architecture.png")
        shutil.copyfile(output_path, dest_brain)
        print(f"Copied to brain artifact directory: {dest_brain}")

if __name__ == "__main__":
    create_perception_tracking_diagram()
