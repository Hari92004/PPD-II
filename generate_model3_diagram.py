"""
Generate Comprehensive Scientific & Visual Diagram for Model 3:
Physics-Based 77 GHz mmWave FMCW Radar Synthesis
Outputs to: D:\PPD2\model3_radar_synthesis_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import shutil

def create_model3_diagram(output_path=r"D:\PPD2\model3_radar_synthesis_architecture.png"):
    fig = plt.figure(figsize=(18, 10), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    # Outer layout with gridspec
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.3, 1.0], height_ratios=[1.0, 1.0],
                          wspace=0.25, hspace=0.32, left=0.05, right=0.96, top=0.88, bottom=0.10)

    # 1. Main Figure Header
    fig.text(0.5, 0.95, "MODEL 3: PHYSICS-BASED 77 GHz mmWAVE FMCW RADAR SYNTHESIS",
             ha='center', va='center', fontsize=18, fontweight='bold', color='#0f172a', fontname='DejaVu Sans')
    fig.text(0.5, 0.915, "Closed-Form Doppler Radar Equations, Swerling RCS Emulation & Hardware Plug-In Architecture",
             ha='center', va='center', fontsize=11, color='#475569', fontname='DejaVu Sans')

    # ==============================================================================
    # PANEL 1 (LEFT, SPANNING ROWS 0-1): GOVERNING WAVE PHYSICS & EQUATIONS
    # ==============================================================================
    ax_eq = fig.add_subplot(gs[:, 0])
    ax_eq.set_facecolor('#f8fafc')
    ax_eq.set_xlim(0, 10)
    ax_eq.set_ylim(0, 10)
    ax_eq.axis('off')

    # Border
    rect_eq = patches.FancyBboxPatch((0.1, 0.1), 9.8, 9.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                                    facecolor='#f8fafc', edgecolor='#cbd5e1', lw=1.5)
    ax_eq.add_patch(rect_eq)

    ax_eq.text(5, 9.5, "1. RADAR GOVERNING PHYSICS", ha='center', va='center',
               fontsize=11, fontweight='bold', color='#0f172a')
    ax_eq.plot([0.5, 9.5], [9.1, 9.1], color='#cbd5e1', lw=1)

    # Chirp Waveform
    ax_eq.text(0.6, 8.6, "A. FMCW Linear Frequency Chirp:", fontsize=9.5, fontweight='bold', color='#0f172a')
    ax_eq.text(0.6, 8.1, "• Carrier Frequency: f0 = 77.0 GHz (λ ≈ 3.89 mm)\n"
                          "• Sweep Bandwidth: B = 4.0 GHz\n"
                          "• Chirp Ramp Slope: S = B / Tc = 1.0 × 10¹⁴ Hz/s", fontsize=8.5, color='#334155')

    # Beat Frequency Box
    box_b = patches.FancyBboxPatch((0.5, 6.7), 9.0, 1.1, boxstyle="round,pad=0.1",
                                  facecolor='#ffffff', edgecolor='#94a3b8', lw=1)
    ax_eq.add_patch(box_b)
    ax_eq.text(5, 7.4, "Range: R = (c · fb) / (2S)", ha='center', va='center',
               fontsize=9, fontweight='bold', color='#0f172a', fontname='DejaVu Sans')
    ax_eq.text(5, 6.95, "Range Resolution: ΔR = c / (2B) = 3.75 cm", ha='center', va='center',
               fontsize=8, color='#0284c7', fontname='DejaVu Sans')

    # Doppler Velocity
    ax_eq.text(0.6, 6.3, "B. Relative Radial Velocity (Doppler):", fontsize=9.5, fontweight='bold', color='#0f172a')
    ax_eq.text(0.6, 5.8, "• Instantaneous Line-of-Sight projection:\n"
                          "  vr = v_rel · u_LOS = (Δv · [x, y]^T) / √(x² + y²)", fontsize=8.5, color='#334155')

    box_d = patches.FancyBboxPatch((0.5, 4.5), 9.0, 1.0, boxstyle="round,pad=0.1",
                                  facecolor='#ffffff', edgecolor='#94a3b8', lw=1)
    ax_eq.add_patch(box_d)
    ax_eq.text(5, 5.15, "Doppler Shift: fd = (2 · vr · f0) / c", ha='center', va='center',
               fontsize=9, fontweight='bold', color='#0f172a', fontname='DejaVu Sans')
    ax_eq.text(5, 4.75, "Max Unambiguous Velocity: |v_max| = ±52.4 m/s", ha='center', va='center',
               fontsize=8, color='#0284c7', fontname='DejaVu Sans')

    # Radar Range Equation
    ax_eq.text(0.6, 4.0, "C. Radar Range Equation & Power:", fontsize=9.5, fontweight='bold', color='#0f172a')
    box_r = patches.FancyBboxPatch((0.5, 2.7), 9.0, 1.1, boxstyle="round,pad=0.1",
                                  facecolor='#ffffff', edgecolor='#94a3b8', lw=1)
    ax_eq.add_patch(box_r)
    ax_eq.text(5, 3.4, "Pr = (Pt · G² · λ² · σ) / ((4π)³ · R⁴ · L)", ha='center', va='center',
               fontsize=9, fontweight='bold', color='#0f172a', fontname='DejaVu Sans')
    ax_eq.text(5, 2.95, "Swerling-I/II Fluctuating RCS (σ in m²)", ha='center', va='center',
               fontsize=8, color='#334155', fontname='DejaVu Sans')

    # Noise injection
    ax_eq.text(0.6, 2.2, "D. Industrial Sensor Noise (TI AWR1843):", fontsize=9.5, fontweight='bold', color='#0f172a')
    ax_eq.text(0.6, 1.3, "• Range Noise: σ_R = 0.04 m (Gaussian)\n"
                          "• Doppler Noise: σ_v = 0.12 m/s\n"
                          "• Azimuth Bearing Jitter: σ_θ = 0.85°\n"
                          "• Multi-target clutter: 60-90 returns/scan", fontsize=8.5, color='#334155')

    # ==============================================================================
    # PANEL 2 (CENTER, SPANNING ROWS 0-1): POLAR RADAR SCOPE SIMULATION
    # ==============================================================================
    ax_scope = fig.add_subplot(gs[:, 1])
    ax_scope.set_facecolor('#07090e') # High-tech dark cockpit scope
    ax_scope.set_aspect('equal')
    ax_scope.set_xlim(-30, 30)
    ax_scope.set_ylim(-2, 58)

    # Range Rings (10m to 50m)
    theta = np.linspace(-np.pi/4, np.pi/4, 200) # +/- 45 deg FOV
    for r in [10, 20, 30, 40, 50]:
        x_ring = r * np.sin(theta)
        y_ring = r * np.cos(theta)
        ax_scope.plot(x_ring, y_ring, color='#1e293b', lw=1.2, ls='--')
        ax_scope.text(0, r + 0.8, f"{r}m", color='#38bdf8', fontsize=7.5, ha='center', fontweight='bold')

    # Azimuth FOV Rays (+/- 45 deg, +/- 22.5 deg, 0 deg)
    for ang in [-np.pi/4, -np.pi/8, 0, np.pi/8, np.pi/4]:
        ax_scope.plot([0, 52 * np.sin(ang)], [0, 52 * np.cos(ang)], color='#1e293b', lw=0.8)

    # Simulated Clutter & Noise Returns (Gray/cyan dots)
    np.random.seed(42)
    clutter_r = np.random.uniform(5, 52, 70)
    clutter_th = np.random.uniform(-np.pi/4, np.pi/4, 70)
    ax_scope.scatter(clutter_r * np.sin(clutter_th), clutter_r * np.cos(clutter_th),
                     s=8, c='#334155', alpha=0.5, label='Ambient Clutter')

    # Dynamic Obstacles with Doppler Velocity Vectors
    # 1. Car (Sky Blue, Approaching vr = -6.2 m/s -> Red Vector)
    ax_scope.scatter([5.5], [24.0], s=80, c='#38bdf8', edgecolors='#ffffff', lw=1.5, zorder=5)
    ax_scope.arrow(5.5, 24.0, -0.6, -4.5, head_width=1.2, head_length=1.5, fc='#ef4444', ec='#ef4444', lw=1.5, zorder=6)
    ax_scope.text(7.0, 24.0, "Car (RCS: 12 dBsm)\nvr: -6.2 m/s (Closing)", color='#ffffff', fontsize=7.5, fontweight='bold')

    # 2. Heavy Truck (Orange, Approaching vr = -4.1 m/s)
    ax_scope.scatter([-8.5], [38.0], s=130, c='#f97316', edgecolors='#ffffff', lw=1.5, zorder=5)
    ax_scope.arrow(-8.5, 38.0, 0.4, -4.0, head_width=1.2, head_length=1.5, fc='#ef4444', ec='#ef4444', lw=1.5, zorder=6)
    ax_scope.text(-7.0, 39.5, "Truck (RCS: 16 dBsm)\nvr: -4.1 m/s", color='#ffffff', fontsize=7.5, fontweight='bold')

    # 3. Two-Wheeler (Purple, Receding vr = +2.5 m/s -> Green Vector)
    ax_scope.scatter([-3.2], [14.0], s=60, c='#a855f7', edgecolors='#ffffff', lw=1.5, zorder=5)
    ax_scope.arrow(-3.2, 14.0, -0.2, 3.0, head_width=1.0, head_length=1.2, fc='#22c55e', ec='#22c55e', lw=1.5, zorder=6)
    ax_scope.text(-12.5, 14.0, "Bike (RCS: 4 dBsm)\nvr: +2.5 m/s", color='#ffffff', fontsize=7.5, fontweight='bold')

    # 4. Pedestrian (Green, Transverse vr = -0.4 m/s)
    ax_scope.scatter([8.2], [12.0], s=40, c='#4ade80', edgecolors='#ffffff', lw=1.2, zorder=5)
    ax_scope.text(9.5, 12.0, "Pedestrian\nRCS: 0.8 m²", color='#ffffff', fontsize=7, fontweight='bold')

    # Ego Vehicle at Bottom
    ego_box = patches.Rectangle((-1.2, -1.5), 2.4, 3.0, fc='#0284c7', ec='#38bdf8', lw=1.5, zorder=10)
    ax_scope.add_patch(ego_box)
    ax_scope.arrow(0, 1.6, 0, 3.0, head_width=1.0, head_length=1.2, fc='#ffffff', ec='#ffffff', lw=1.2, zorder=11)
    ax_scope.text(0, -3.2, "EGO VEHICLE", color='#94a3b8', fontsize=7.5, ha='center', fontweight='bold')

    # Scope Header
    ax_scope.text(0, 54, "77 GHz RADAR POLAR BEV SCOPE", color='#ffffff', fontsize=9.5, ha='center', fontweight='bold')
    ax_scope.text(-28, 54, "FOV: 90°", color='#64748b', fontsize=7.5)
    ax_scope.text(28, 54, "Rmax: 55m", color='#64748b', fontsize=7.5, ha='right')

    ax_scope.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax_scope.spines.values():
        spine.set_color('#334155')

    # ==============================================================================
    # PANEL 3 (TOP RIGHT): RANGE-DOPPLER 2D SPECTRUM
    # ==============================================================================
    ax_rd = fig.add_subplot(gs[0, 2])
    ax_rd.set_facecolor('#0f172a')
    
    # Generate authentic synthetic Range-Doppler 2D map
    ranges = np.linspace(0, 55, 100)
    velocities = np.linspace(-25, 25, 100)
    R_grid, V_grid = np.meshgrid(ranges, velocities)
    
    # Base thermal noise floor
    RD_map = -80 + np.random.normal(0, 3, R_grid.shape)
    
    # Target 1: Car peak at R=24m, V=-6.2 m/s
    RD_map += 45 * np.exp(-((R_grid - 24)**2 / 6 + (V_grid - (-6.2))**2 / 3))
    # Target 2: Truck peak at R=38m, V=-4.1 m/s
    RD_map += 52 * np.exp(-((R_grid - 38)**2 / 10 + (V_grid - (-4.1))**2 / 4))
    # Target 3: Bike peak at R=14m, V=+2.5 m/s
    RD_map += 32 * np.exp(-((R_grid - 14)**2 / 4 + (V_grid - 2.5)**2 / 2))

    im = ax_rd.imshow(RD_map, extent=[0, 55, -25, 25], aspect='auto', origin='lower', cmap='inferno')
    ax_rd.set_title("Range-Doppler 2D Spectrum", fontsize=9, fontweight='bold', color='#0f172a', pad=6)
    ax_rd.set_xlabel("Range (meters)", fontsize=8, color='#334155')
    ax_rd.set_ylabel("Radial Velocity (m/s)", fontsize=8, color='#334155')
    ax_rd.tick_params(labelsize=7, colors='#334155')
    cbar = fig.colorbar(im, ax=ax_rd, fraction=0.046, pad=0.04)
    cbar.set_label("Power (dBm)", fontsize=7, color='#334155')
    cbar.ax.tick_params(labelsize=6, colors='#334155')

    # ==============================================================================
    # PANEL 4 (BOTTOM RIGHT): SWERLING RCS CALIBRATION BENCHMARKS
    # ==============================================================================
    ax_rcs = fig.add_subplot(gs[1, 2])
    ax_rcs.set_facecolor('#ffffff')
    
    classes = ['Pedestrian', 'Two-Wheeler', 'Auto-Rickshaw', 'Passenger Car', 'Heavy Truck']
    rcs_vals = [0.8, 3.5, 6.0, 15.0, 40.0] # in m^2
    rcs_dbsm = [10 * np.log10(v) for v in rcs_vals]
    
    bar_colors = ['#4ade80', '#a855f7', '#06b6d4', '#38bdf8', '#f97316']
    bars = ax_rcs.barh(classes, rcs_dbsm, color=bar_colors, edgecolor='#0f172a', lw=1, height=0.55)
    
    ax_rcs.set_title("Empirical Radar Cross-Section (RCS)", fontsize=9, fontweight='bold', color='#0f172a', pad=6)
    ax_rcs.set_xlabel("RCS Value in dBsm", fontsize=8, color='#334155')
    ax_rcs.tick_params(labelsize=7.5, colors='#334155')
    ax_rcs.set_xlim(-5, 20)
    ax_rcs.grid(axis='x', ls='--', alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax_rcs.text(w + 0.6, bar.get_y() + bar.get_height()/2, f"{w:.1f} dBsm",
                    va='center', fontsize=7, fontweight='bold', color='#0f172a')

    for spine in ax_rcs.spines.values():
        spine.set_color('#cbd5e1')

    # ==============================================================================
    # BOTTOM BANNER: HARDWARE PLUG-IN & DEPLOYMENT BRIDGE
    # ==============================================================================
    banner_rect = patches.FancyBboxPatch((0.05, 0.015), 0.91, 0.055, boxstyle="round,pad=0.01,rounding_size=0.01",
                                         facecolor='#0f172a', edgecolor='none', transform=fig.transFigure)
    fig.patches.append(banner_rect)
    
    fig.text(0.5, 0.042, "VIRTUAL-TO-PHYSICAL HARDWARE BRIDGE: PROTOTYPED VIA CLOSED-FORM PHYSICS  ➔  PRODUCTION VEHICLE PLUG-IN VIA CAN BUS",
             ha='center', va='center', fontsize=9.5, fontweight='bold', color='#ffffff', fontname='DejaVu Sans')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Model 3 Architecture Diagram saved: {output_path}")

    # Copy to artifacts for immediate inline viewing
    art_dir = r"C:\Users\harip\.gemini\antigravity\brain\2d6e9244-0990-4251-8a0a-dd1256a14492"
    art_path = os.path.join(art_dir, "model3_radar_synthesis_architecture.png")
    shutil.copy(output_path, art_path)
    print(f"[+] Copied to artifacts: {art_path}")

if __name__ == "__main__":
    create_model3_diagram()
