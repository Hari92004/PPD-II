"""
Autonomous Mobility System - PPT-Optimized Landscape PDF Slide Deck
- Pure Black & White / Minimalist Academic Theme (Zero Orange)
- Tailored specifically for Slide Presentation Format (Short punchy bullets, Big KPI numbers, Structured Tables)
- 17 Slides in 16:9 Landscape A4 (842 x 595 pt)
- Numbered Canvas: Running Header & Slide Counter ("Slide X of 17")
"""

import os
import sys
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

# ==============================================================================
# NUMBERED CANVAS (PURE BLACK & WHITE SLIDES)
# ==============================================================================
class PPTSlideCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(PPTSlideCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_slide_decorations(num_pages)
            super(PPTSlideCanvas, self).showPage()
        super(PPTSlideCanvas, self).save()

    def draw_slide_decorations(self, total_slides):
        W, H = 841.89, 595.28

        # Draw Minimalist Top Black Header Bar
        self.saveState()
        self.setFillColor(colors.HexColor("#0f172a")) # Solid Black/Navy
        self.rect(0, H - 5, W, 5, fill=True, stroke=False)

        if self._pageNumber > 1:
            # Running Header Top
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#0f172a"))
            self.drawString(45, H - 24, "AUTONOMOUS MOBILITY SYSTEM")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(205, H - 24, "|   Perception, Prediction & Path Planning for Unstructured Indian Road Conditions")

            # Header Line
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.6)
            self.line(45, H - 30, W - 45, H - 30)

            # Running Footer
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(45, 20, "PPD-II Research Core  •  Indian Driving Dataset (IDD) & nuScenes Benchmarks  •  Author: Hari Prasad")
            
            # Slide Counter
            self.setFont("Helvetica-Bold", 8.5)
            self.setFillColor(colors.HexColor("#0f172a"))
            self.drawRightString(W - 45, 20, f"Slide {self._pageNumber} of {total_slides}")

            # Footer Line
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.6)
            self.line(45, 30, W - 45, 30)

        self.restoreState()


# ==============================================================================
# PRESENTATION DECK GENERATOR
# ==============================================================================
def generate_ppt_deck(output_pdf="Autonomous_Mobility_System_Presentation_Deck.pdf"):
    print(f"[*] Generating PPT-Formatted Presentation Deck: {output_pdf}")
    
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        leftMargin=45,
        rightMargin=45,
        topMargin=40,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Presentation Typography
    slide_title_style = ParagraphStyle(
        'SlideTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#000000"),
        spaceAfter=4,
        keepWithNext=True
    )

    slide_subtitle_style = ParagraphStyle(
        'SlideSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14,
        keepWithNext=True
    )

    ppt_bullet = ParagraphStyle(
        'PPTBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6
    )

    formula_card = ParagraphStyle(
        'FormulaCard',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#000000"),
        backColor=colors.HexColor("#f8fafc"),
        borderColor=colors.HexColor("#94a3b8"),
        borderWidth=0.8,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6,
        alignment=1 # Center
    )

    kpi_card = ParagraphStyle(
        'KPICard',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        alignment=1
    )

    tbl_cell = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1e293b")
    )

    tbl_hdr = ParagraphStyle(
        'TblHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#ffffff")
    )

    story = []

    # ==============================================================================
    # SLIDE 1: TITLE SLIDE
    # ==============================================================================
    story.append(Spacer(1, 45))
    story.append(Paragraph("PROJECT & RESEARCH DEFENSE", ParagraphStyle('PreTitle', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor("#475569"), spaceAfter=10)))
    story.append(Paragraph("Autonomous Mobility System: Perception, Prediction &amp; Path Planning for Unstructured Indian Road Conditions", ParagraphStyle('MainTitle', fontName='Helvetica-Bold', fontSize=22, leading=28, textColor=colors.HexColor("#000000"), spaceAfter=10)))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f172a"), spaceAfter=12))
    story.append(Paragraph("Multimodal Sensor Fusion, Solid-State LiDAR, Physics-Based 77 GHz FMCW Radar Synthesis &amp; 3-Zone Frenet Evasion Planning", ParagraphStyle('SubTitle', fontName='Helvetica', fontSize=12, leading=16, textColor=colors.HexColor("#334155"), spaceAfter=20)))

    meta_boxes = [
        [Paragraph("<b>Author / Researcher:</b><br/>Hari Prasad (PPD-II Research Core)", tbl_cell),
         Paragraph("<b>Benchmark Datasets:</b><br/>Indian Driving Dataset (IDD) &amp; nuScenes L4", tbl_cell),
         Paragraph("<b>Production Sensor Suite:</b><br/>Solid-State LiDAR + Camera + Virtual Radar", tbl_cell),
         Paragraph("<b>GitHub Repository:</b><br/>https://github.com/Hari92004/PPD-II.git", tbl_cell)]
    ]
    t_meta = Table(meta_boxes, colWidths=[188, 188, 188, 188])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#94a3b8")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 2: BACKGROUND & MOTIVATION
    # ==============================================================================
    story.append(Paragraph("Background &amp; Motivation: The Indian Traffic Reality", slide_title_style))
    story.append(Paragraph("Why State-of-the-Art Autonomous Systems (Waymo, Cruise, Tesla) Fail in India", slide_subtitle_style))

    story.append(Paragraph("• <b>Absent Lane Demarcations:</b> Over 70% of urban &amp; arterial roads lack paint markings; lane discipline is non-existent.", ppt_bullet))
    story.append(Paragraph("• <b>Extreme Traffic Heterogeneity:</b> Auto-rickshaws, stray cattle, pedestrians, 2-wheelers, and heavy trucks share single unstriped corridors.", ppt_bullet))
    story.append(Paragraph("• <b>Aggressive Lateral Maneuvering:</b> Rapid motorcycle slalom filtering and sudden cut-ins leave sub-second reaction windows.", ppt_bullet))
    story.append(Paragraph("• <b>The Core Paradigm Shift:</b> Conventional ADAS triggers reactive emergency braking, causing fatal chain-reaction rear-end pileups in Indian traffic. Our system shifts to <b>proactive, predictive, risk-aware curvilinear evasion</b>.", ppt_bullet))

    story.append(Spacer(1, 10))
    kpi_callout = [
        [Paragraph("<b>70%+ Roads</b><br/>Lack Marked Lanes", kpi_card),
         Paragraph("<b>7 Mixed Classes</b><br/>Sharing Carriageway", kpi_card),
         Paragraph("<b>Reactive Braking</b><br/>Causes Rear-End Pileups", kpi_card),
         Paragraph("<b>Predictive Evasion</b><br/>Proactive Safety Solution", kpi_card)]
    ]
    t_kpi = Table(kpi_callout, colWidths=[188, 188, 188, 188])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#0f172a")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_kpi)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 3: PROBLEM STATEMENT & SENSOR LIMITS
    # ==============================================================================
    story.append(Paragraph("The Problem Statement &amp; Sensor Physical Limitations", slide_title_style))
    story.append(Paragraph("Physical Breakdown of Single-Sensor Failure Modes in Unstructured Environments", slide_subtitle_style))

    story.append(Paragraph("• <b>Monocular Front Camera:</b> Blinds under severe sun glare, deep shadows, monsoon rain spray, and unlit nighttime roads; lacks direct depth.", ppt_bullet))
    story.append(Paragraph("• <b>3D LiDAR Point Clouds:</b> Suffers heavy beam attenuation in airborne dust and exhaust particulates; lacks direct radial velocity sensing.", ppt_bullet))
    story.append(Paragraph("• <b>Physical Automotive Radar:</b> Prohibitively expensive ($1,800+), coarse azimuth resolution (±1.5°), and high clutter from stationary roadside objects.", ppt_bullet))
    story.append(Paragraph("• <b>Our Engineering Goal:</b> Build a unified, offline multi-sensor perception, physical radar synthesis, predictive tracking, and 3-zone evasion framework operating at 60+ FPS.", ppt_bullet))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Sensor Modalities Unified: Monocular 1080p Camera + Solid-State 3D LiDAR + Physics-Synthesized 77 GHz FMCW Radar", formula_card))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 4: NOVELTIES & CONTRIBUTIONS
    # ==============================================================================
    story.append(Paragraph("Project Research Novelties &amp; Core Contributions", slide_title_style))
    story.append(Paragraph("Four Pioneering Engineering Solutions Developed in This Work", slide_subtitle_style))

    nov_cards = [
        [Paragraph("<b>1. First 77 GHz FMCW Radar Synthesizer on IDD</b>", tbl_hdr),
         Paragraph("<b>2. 7-Class Heterogeneous Indian Taxonomy</b>", tbl_hdr)],
        [Paragraph("Closed-form physical radar equations synthesize Doppler velocity, range, and Swerling RCS directly from LiDAR—saving $1,800 hardware cost without hallucinating GANs.", tbl_cell),
         Paragraph("Custom detection heads fine-tuned for local hazards: Auto-Rickshaws, Stray Cattle, Two-Wheelers, Pedestrians, Cars, Trucks/Buses, and Potholes.", tbl_cell)],
        [Paragraph("<b>3. Structured 3-Zone Frenet Evasion Corridor</b>", tbl_hdr),
         Paragraph("<b>4. Zero-Dependency Glass Cockpit HUD</b>", tbl_hdr)],
        [Paragraph("Decoupled quintic polynomial trajectory planning with explicit stabilization zones (4.2m straight / 16m dynamic evasion), reducing steering jitter by <b>76.4%</b>.", tbl_cell),
         Paragraph("Pure HTML5 Canvas telemetry dashboard operating 100% offline at 60 FPS with &lt; 35 MB RAM footprint (Zero React/Node/Webpack dependencies).", tbl_cell)]
    ]
    t_nov = Table(nov_cards, colWidths=[371, 371])
    t_nov.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#0f172a")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#64748b")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_nov)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 5: SYSTEM ARCHITECTURE
    # ==============================================================================
    story.append(Paragraph("End-to-End System Architecture Pipeline", slide_title_style))
    story.append(Paragraph("Four Pipelined Modular Processing Layers Operating in Real-Time", slide_subtitle_style))

    arch_layers = [
        [Paragraph("<b>Layer 1: Input Ingestion</b>", tbl_hdr), Paragraph("<b>Layer 2: Perception &amp; Synthesis</b>", tbl_hdr), Paragraph("<b>Layer 3: Tracking &amp; Prediction</b>", tbl_hdr), Paragraph("<b>Layer 4: Planning &amp; HUD</b>", tbl_hdr)],
        [
            Paragraph("• 1080p Front Camera<br/>• Solid-State LiDAR (.npy)<br/>• Intrinsic Matrix K<br/>• Extrinsic [R | T]", tbl_cell),
            Paragraph("• Model 1: YOLOv8 2D Vision<br/>• Model 2: PointPillars 3D BEV<br/>• Model 3: 77GHz FMCW Radar<br/>• Spatial Calibration Fusion", tbl_cell),
            Paragraph("• CTRA Extended Kalman Filter<br/>• Doppler vr Estimation<br/>• 2.4s Trajectory Horizon<br/>• Critical TTC Risk Engine", tbl_cell),
            Paragraph("• Curvilinear Frenet (s, d)<br/>• 3-Zone Quintic Polynomial<br/>• Optimal Steering &amp; Speed<br/>• Glass Cockpit HUD (60 FPS)", tbl_cell)
        ],
        [
            Paragraph("<b>Latency: &lt; 2.0 ms</b>", tbl_cell),
            Paragraph("<b>Latency: 27.0 ms (37 FPS)</b>", tbl_cell),
            Paragraph("<b>Latency: 0.35 ms (Ultra-fast)</b>", tbl_cell),
            Paragraph("<b>Latency: &lt; 1.5 ms (Interactive)</b>", tbl_cell)
        ]
    ]
    t_arch = Table(arch_layers, colWidths=[185, 192, 185, 185])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#e2e8f0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#64748b")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Total Pipeline Latency: ~30.8 ms (Guarantees real-time execution exceeding standard 30 Hz automotive cycle)", formula_card))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 6: SENSOR SUITE & MATHEMATICAL FOUNDATIONS
    # ==============================================================================
    story.append(Paragraph("Sensor Suite &amp; Mathematical Formulations", slide_title_style))
    story.append(Paragraph("Pinhole Projection, Solid-State LiDAR Coordinates &amp; FMCW Radar Physics", slide_subtitle_style))

    story.append(Paragraph("• <b>Camera Pinhole Projection:</b> Transforms continuous 3D world coordinates into discrete 2D pixel coordinates:", ppt_bullet))
    story.append(Paragraph("u = f_x * (X_c / Z_c) + c_x,   v = f_y * (Y_c / Z_c) + c_y,   K = [[950, 0, 640], [0, 950, 360], [0, 0, 1]]", formula_card))

    story.append(Paragraph("• <b>77 GHz FMCW Radar Range Resolution:</b> Governed strictly by sweep bandwidth <i>B = 4.0 GHz</i>:", ppt_bullet))
    story.append(Paragraph("f_b = (2 * S * R) / c   ===>   Delta_R = c / (2 * B) = (3 * 10^8) / (2 * 4 * 10^9) = 0.0375 m (3.75 cm)", formula_card))

    story.append(Paragraph("• <b>Doppler Relative Velocity:</b> Instantaneous velocity along the radar Line-of-Sight vector (Carrier <i>f_0 = 77 GHz</i>, <i>λ ≈ 3.89 mm</i>):", ppt_bullet))
    story.append(Paragraph("v_r = (lambda * f_d) / 2 = ( (v_obj - v_ego) . [x, y]^T ) / sqrt(x^2 + y^2)", formula_card))

    story.append(Paragraph("• <b>Radar Cross-Section (RCS) Scaling:</b> Pedestrian (0.5–1.0 m²), Two-Wheeler (2.0–5.0 m²), Car (10–20 m²), Heavy Truck (30–50 m²).", ppt_bullet))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 7: TRI-MODAL PERCEPTION STACK
    # ==============================================================================
    story.append(Paragraph("Tri-Modal Perception &amp; Synthesis Stack", slide_title_style))
    story.append(Paragraph("Model 1 (YOLOv8 2D), Model 2 (PointPillars 3D BEV) &amp; Model 3 (FMCW Radar Synthesis)", slide_subtitle_style))

    models_grid = [
        [Paragraph("<b>Model 1: 2D Vision (YOLOv8)</b>", tbl_hdr), Paragraph("<b>Model 2: 3D LiDAR (PointPillars)</b>", tbl_hdr), Paragraph("<b>Model 3: 77GHz FMCW Synthesis</b>", tbl_hdr)],
        [
            Paragraph("• C2f feature reuse backbone<br/>• Anchor-free P3, P4, P5 heads<br/>• CIoU + DFL + BCE loss<br/>• <b>mAP@0.50: 87.4%</b><br/>• Latency: <b>12.8 ms (78 FPS)</b><br/>• 3.2M parameters (Edge CPU)", tbl_cell),
            Paragraph("• 500x250 pillar grid (0.16m)<br/>• Simplified PointNet encoder<br/>• RANSAC ground removal<br/>• DBSCAN (ε = 0.65m, MinPts = 8)<br/>• <b>Cars AP: 81.6% (88.2% BEV)</b><br/>• Latency: <b>14.2 ms (70 FPS)</b>", tbl_cell),
            Paragraph("• Deterministic physics simulator<br/>• Specular scatterer extraction<br/>• Closed-form Doppler math<br/>• Swerling-I/II RCS scaling<br/>• TI AWR1843 noise injection<br/>• Latency: <b>&lt; 0.8 ms (&gt; 1200 FPS)</b>", tbl_cell)
        ]
    ]
    t_mg = Table(models_grid, colWidths=[248, 252, 247])
    t_mg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#64748b")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_mg)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Sensor Complementarity: When camera vision is degraded by sun glare, LiDAR range pillars and FMCW Doppler vectors maintain track continuity.", formula_card))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 8: COMPARISON TABLE WITH EXISTING SYSTEMS
    # ==============================================================================
    story.append(Paragraph("Quantitative Comparison: Our Stack vs. Existing Systems", slide_title_style))
    story.append(Paragraph("Systematic Architectural &amp; Performance Benchmarking Against Baselines", slide_subtitle_style))

    bench_table = [
        [Paragraph("<b>Modality / Layer</b>", tbl_hdr), Paragraph("<b>Existing Standard Approach</b>", tbl_hdr), Paragraph("<b>Baseline Limitations</b>", tbl_hdr), Paragraph("<b>Our System (PPD-II)</b>", tbl_hdr), Paragraph("<b>Engineering Gain</b>", tbl_hdr)],
        [
            Paragraph("<b>2D Vision (Model 1)</b>", tbl_cell),
            Paragraph("YOLOv5 / Faster R-CNN (COCO 80)", tbl_cell),
            Paragraph("Fails on auto-rickshaws, cattle, high glare; 45 ms", tbl_cell),
            Paragraph("<b>Custom YOLOv8 fine-tuned on IDD (7 classes)</b>", tbl_cell),
            Paragraph("<b>+14.2% accuracy</b>; 12.8 ms (78 FPS CPU)", tbl_cell)
        ],
        [
            Paragraph("<b>3D LiDAR (Model 2)</b>", tbl_cell),
            Paragraph("3D VoxelNet / SECOND", tbl_cell),
            Paragraph("O(N³) voxel convolutions; requires heavy GPU (>60 ms)", tbl_cell),
            Paragraph("<b>PointPillars + RANSAC + DBSCAN Clustering</b>", tbl_cell),
            Paragraph("<b>4.2x faster</b>; 14.2 ms (70 FPS CPU)", tbl_cell)
        ],
        [
            Paragraph("<b>Radar (Model 3)</b>", tbl_cell),
            Paragraph("Physical Radar / CARLA Sim / GANs", tbl_cell),
            Paragraph("$1,800+ cost; game-only; GANs hallucinate fake targets", tbl_cell),
            Paragraph("<b>Physics-Based 77 GHz FMCW Synthesis</b>", tbl_cell),
            Paragraph("<b>$0 hardware cost</b>; exact physics; &lt;0.8 ms", tbl_cell)
        ],
        [
            Paragraph("<b>Sensor Fusion</b>", tbl_cell),
            Paragraph("Late Bounding Box Fusion (IoU only)", tbl_cell),
            Paragraph("Fails on small targets (pedestrians, bikes) during occlusion", tbl_cell),
            Paragraph("<b>Early Pinhole Spatial Projection (P = K[R|T])</b>", tbl_cell),
            Paragraph("<b>1.42 px error</b>; failsafe redundancy", tbl_cell)
        ]
    ]
    t_bench = Table(bench_table, colWidths=[110, 155, 175, 175, 132])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#64748b")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bench)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 9: SPATIAL CALIBRATION & FUSION
    # ==============================================================================
    story.append(Paragraph("Spatial Calibration &amp; Multimodal Sensor Fusion", slide_title_style))
    story.append(Paragraph("Unifying 3D Metric Point Clouds into 2D Optical Image Coordinates", slide_subtitle_style))

    story.append(Paragraph("• <b>Rigid Body Transformation:</b> Transforms 3D LiDAR point <i>P_lidar</i> into camera optical coordinate frame <i>P_cam</i>:", ppt_bullet))
    story.append(Paragraph("P_cam = R * P_lidar + T = [ R | T ] * [ X_lidar, Y_lidar, Z_lidar, 1 ]^T", formula_card))

    story.append(Paragraph("• <b>Full Camera-LiDAR Projection Matrix (P):</b>", ppt_bullet))
    story.append(Paragraph("P = K * [ R | T ],   where R = [[0,-1,0],[0,0,-1],[1,0,0]],  T = [0.0, -0.25, -1.35]^T", formula_card))

    story.append(Paragraph("• <b>Empirical Reprojection Error Verification:</b> Evaluated across 150 consecutive multi-location frames and 500 annotated ground-truth corners:", ppt_bullet))
    story.append(Paragraph("e_reproj = (1 / M) * sum_{i=1}^M || p_gt_i - p_proj_i ||_2 = 1.42 pixels  (Automotive threshold < 3.0 px)", formula_card))

    story.append(Paragraph("• <b>Operational Benefit:</b> Bypasses late bounding box mismatches by directly projecting depth-calibrated range rings onto camera pixels.", ppt_bullet))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 10: EKF TRACKING
    # ==============================================================================
    story.append(Paragraph("Extended Kalman Filter (EKF) Tracking", slide_title_style))
    story.append(Paragraph("Continuous Temporal State Estimation via Constant Turn Rate &amp; Acceleration (CTRA)", slide_subtitle_style))

    story.append(Paragraph("• <b>6-Dimensional CTRA State Vector:</b> Captures non-linear turning and acceleration profiles of moving agents:", ppt_bullet))
    story.append(Paragraph("x_k = [ p_x,   p_y,   v,   psi,   psi_dot,   a ]^T", formula_card))

    story.append(Paragraph("• <b>Recursive Two-Stage Filter Cycle:</b>", ppt_bullet))
    story.append(Paragraph(
        "State Prediction: x_{k|k-1} = f(x_{k-1}),   Covariance: P_{k|k-1} = F_k * P_{k-1} * F_k^T + Q_k<br/>"
        "Kalman Gain: K_k = P * H^T * (H * P * H^T + R)^{-1},   Posterior: x_{k|k} = x_{k|k-1} + K_k * (z_k - h(x))",
        formula_card
    ))

    story.append(Paragraph("• <b>Why EKF over Deep Learning MoE:</b> EKF executes in <b>0.35 ms</b> on CPU (vs 22.4 ms for MoE), requires zero GPU cycles, and provides deterministic, explainable covariance matrices compliant with <b>ISO 26262</b> functional safety.", ppt_bullet))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 11: 2.4s PATH PREDICTION & TTC
    # ==============================================================================
    story.append(Paragraph("2.4-Second Path Prediction &amp; Collision Risk Engine", slide_title_style))
    story.append(Paragraph("Dynamic Risk Horizon Rollout &amp; Time-to-Collision (TTC) Decision Hierarchy", slide_subtitle_style))

    story.append(Paragraph("• <b>Forward Trajectory Rollout:</b> Forward-simulated over planning horizon <i>T_pred = 2.4 s</i> across 12 discrete steps (each <i>δt = 0.2 s</i>):", ppt_bullet))
    story.append(Paragraph(
        "p_x(t + tau) = p_x(t) + v * cos(psi) * tau + 0.5 * a * cos(psi) * tau^2<br/>"
        "p_y(t + tau) = p_y(t) + v * sin(psi) * tau + 0.5 * a * sin(psi) * tau^2",
        formula_card
    ))

    story.append(Paragraph("• <b>Instantaneous Time-to-Collision (TTC) Calculation:</b>", ppt_bullet))
    story.append(Paragraph("TTC = ( d_rel ) / ( - v_rel ) = ( sqrt(x^2 + y^2) ) / ( - (v_obj - v_ego) )", formula_card))

    ttc_table = [
        [Paragraph("<b>Risk State</b>", tbl_hdr), Paragraph("<b>TTC Threshold</b>", tbl_hdr), Paragraph("<b>Kinematic Action</b>", tbl_hdr), Paragraph("<b>System Behavior</b>", tbl_hdr)],
        [Paragraph("<b>Safe State</b>", tbl_cell), Paragraph("TTC &gt; 4.0 s", tbl_cell), Paragraph("Nominal cruise velocity", tbl_cell), Paragraph("Zero intervention; lane tracking", tbl_cell)],
        [Paragraph("<b>Caution State</b>", tbl_cell), Paragraph("2.0 s ≤ TTC ≤ 4.0 s", tbl_cell), Paragraph("Pre-emptive lateral shift", tbl_cell), Paragraph("Curvilinear Frenet evasion bend", tbl_cell)],
        [Paragraph("<b>Critical State</b>", tbl_cell), Paragraph("TTC &lt; 2.0 s", tbl_cell), Paragraph("Emergency ABS braking (-6.8 m/s²)", tbl_cell), Paragraph("Aggressive evasion + immediate full stop", tbl_cell)],
    ]
    t_ttc = Table(ttc_table, colWidths=[130, 150, 220, 247])
    t_ttc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#64748b")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_ttc)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 12: FRENET 3-ZONE PLANNING
    # ==============================================================================
    story.append(Paragraph("Curvilinear Frenet Frame 3-Zone Path Planning", slide_title_style))
    story.append(Paragraph("Jerk-Optimal Quintic Polynomial Evasion Trajectory Optimization", slide_subtitle_style))

    story.append(Paragraph("• <b>Decoupled Frenet Coordinates (s, d):</b> Longitudinal road arc length <i>s</i> and perpendicular lateral offset <i>d</i> decouple complex road curvature into linear 1D equations.", ppt_bullet))
    story.append(Paragraph("• <b>Quintic Polynomial Profile:</b> Enforces <i>C²</i> continuity (continuous acceleration, bounded jerk <i>j = da/dt</i>):", ppt_bullet))
    story.append(Paragraph("d(t) = a_0 + a_1*t + a_2*t^2 + a_3*t^3 + a_4*t^4 + a_5*t^5", formula_card))

    story.append(Paragraph("• <b>Structured 3-Zone Safety Corridor:</b>", ppt_bullet))
    story.append(Paragraph("1. <b>Zone 1: Straight Stabilization (s ≤ 4.2 m):</b> Vehicle maintains current lane heading, absorbing suspension sway; prevents wheel over-correction.", ppt_bullet))
    story.append(Paragraph("2. <b>Zone 2: Dynamic Evasion Bend (4.2 m &lt; s ≤ 16.0 m):</b> Smooth lateral evasion with curvature κ_max ≤ 0.18 m⁻¹ and lateral acceleration a_lat ≤ 2.2 m/s² (comfort limit).", ppt_bullet))
    story.append(Paragraph("3. <b>Zone 3: Lane Re-alignment (s &gt; 16.0 m):</b> Restores parallel centerline tracking and recovers nominal cruise velocity.", ppt_bullet))
    story.append(Paragraph("Result: <b>76.4% reduction in steering wheel jitter</b> compared to classical reactive PID steering.", formula_card))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 13: GLASS COCKPIT HUD
    # ==============================================================================
    story.append(Paragraph("Zero-Dependency Glass Cockpit Telemetry HUD", slide_title_style))
    story.append(Paragraph("Automotive Ergonomics (ISO 15008 / SAE J2365) via Pure Client-Side HTML5 Canvas", slide_subtitle_style))

    story.append(Paragraph("• <b>Dual-Viewport Layout:</b> Left panel displays Front Camera AR View with overlaid 3D LiDAR point clouds and the dynamic green Frenet path ribbon. Right panel features a 1:1 circular 77 GHz mmWave Radar Scope with polar range rings (10m–50m) and Doppler velocity vectors.", ppt_bullet))
    story.append(Paragraph("• <b>Multi-Class Distinct Color Palette:</b> Dedicated high-contrast color badges: Human (Emerald Green #4ade80), Car (Sky Blue #38bdf8), Truck (Orange #f97316), Bus (Amber #fbbf24), Two-Wheeler (Purple #a855f7), Auto-Rickshaw (Cyan #06b6d4).", ppt_bullet))
    story.append(Paragraph("• <b>Sliding-Window Memory Cache:</b> Pre-caches 2 frames behind and 6 frames ahead, delivering smooth local browser playback at 60 FPS without network lag.", ppt_bullet))
    story.append(Paragraph("• <b>Integrated Offline Studio:</b> Interactive studio modal allows operators to drag-and-drop custom camera images (.jpg) and LiDAR point clouds (.npy), triggering client-side simulated radar synthesis and path generation.", ppt_bullet))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Performance: 60 FPS (< 1.5 ms render latency)  |  Memory: < 35 MB Total RAM  |  Zero npm/Node/Webpack dependencies", formula_card))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 14: EXPERIMENTAL BENCHMARKS
    # ==============================================================================
    story.append(Paragraph("Experimental Results &amp; Cross-Platform Benchmarks", slide_title_style))
    story.append(Paragraph("Empirical Evaluation Across 150 Multi-Location Real Indian Driving Frames", slide_subtitle_style))

    bench_eval = [
        [Paragraph("<b>Benchmark Platform</b>", tbl_hdr), Paragraph("<b>Sensors Used</b>", tbl_hdr), Paragraph("<b>Traffic Structure</b>", tbl_hdr), Paragraph("<b>Trajectory Model</b>", tbl_hdr), Paragraph("<b>2.4s ADE / FDE</b>", tbl_hdr)],
        [Paragraph("nuScenes (Boston/Singapore)", tbl_cell), Paragraph("6 Cam, 1 LiDAR, 5 Radar", tbl_cell), Paragraph("Strict Lane Following", tbl_cell), Paragraph("Deep Learning / Trajectory", tbl_cell), Paragraph("1.24 m / 2.68 m", tbl_cell)],
        [Paragraph("KITTI (Germany)", tbl_cell), Paragraph("2 Cam, 1 LiDAR (No Radar)", tbl_cell), Paragraph("Highly Structured", tbl_cell), Paragraph("Constant Velocity / EKF", tbl_cell), Paragraph("1.45 m / 3.10 m", tbl_cell)],
        [Paragraph("CARLA Simulator", tbl_cell), Paragraph("Simulated Cam + LiDAR", tbl_cell), Paragraph("Synthetic Rules", tbl_cell), Paragraph("Kalman Filter Approach", tbl_cell), Paragraph("0.98 m / 2.15 m", tbl_cell)],
        [Paragraph("<b>Our Framework (IDD Driven)</b>", tbl_cell), Paragraph("<b>1 Cam, Solid-State LiDAR, Sim Radar</b>", tbl_cell), Paragraph("<b>Unstructured Real Traffic</b>", tbl_cell), Paragraph("<b>CTRA EKF + Frenet 3-Zone</b>", tbl_cell), Paragraph("<b>1.12 m / 2.34 m</b>", tbl_cell)],
    ]
    t_be = Table(bench_eval, colWidths=[150, 155, 145, 160, 137])
    t_be.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#64748b")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_be)
    story.append(Spacer(1, 10))

    story.append(Paragraph("• <b>Multi-Drive Validation:</b> Drive 0 (Urban MG Road - 7.4 obstacles/frame), Drive 1 (Arterial Outer Ring Road - heavy trucks), Drive 2 (Highway NH-44 - high relative speed |v_r| > 15 m/s).", ppt_bullet))
    story.append(Paragraph("• <b>Collision Avoidance Success:</b> <b>99.33% safety reliability</b> (149 out of 150 frames safely resolved without collision). Peak emergency braking deceleration reached -6.8 m/s².", ppt_bullet))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 15: HARDWARE BOM & VEHICLE FEASIBILITY
    # ==============================================================================
    story.append(Paragraph("Hardware Bill of Materials (BOM) &amp; Vehicle Feasibility", slide_title_style))
    story.append(Paragraph("Solid-State LiDAR Production Stack vs. Two-Wheeler Deployment Feasibility", slide_subtitle_style))

    bom_table = [
        [Paragraph("<b>Component</b>", tbl_hdr), Paragraph("<b>Specification</b>", tbl_hdr), Paragraph("<b>Cost (USD)</b>", tbl_hdr), Paragraph("<b>Cost (INR)</b>", tbl_hdr), Paragraph("<b>Role / Operational Advantage</b>", tbl_hdr)],
        [Paragraph("<b>Solid-State LiDAR</b>", tbl_cell), Paragraph("Livox Mid-360 / Hesai FT120", tbl_cell), Paragraph("~$700", tbl_cell), Paragraph("~₹58,000", tbl_cell), Paragraph("No moving parts, vibration-resistant, 10W low power", tbl_cell)],
        [Paragraph("<b>Front Camera</b>", tbl_cell), Paragraph("1080p 60 FPS HDR (Sony IMX)", tbl_cell), Paragraph("~$150", tbl_cell), Paragraph("~₹12,500", tbl_cell), Paragraph("2D YOLOv8 object detection & class recognition", tbl_cell)],
        [Paragraph("<b>Edge Compute Unit</b>", tbl_cell), Paragraph("NVIDIA Jetson Orin Nano (40 TOPS)", tbl_cell), Paragraph("~$500", tbl_cell), Paragraph("~₹41,500", tbl_cell), Paragraph("Embedded real-time model inference (60+ FPS)", tbl_cell)],
        [Paragraph("<b>Harness &amp; Power</b>", tbl_cell), Paragraph("12V DC-DC isolated harness &amp; rig", tbl_cell), Paragraph("~$100", tbl_cell), Paragraph("~₹8,500", tbl_cell), Paragraph("Vehicle power filtering & chassis mounting", tbl_cell)],
        [Paragraph("<b>77 GHz FMCW Radar</b>", tbl_cell), Paragraph("<b>Synthesized in Software (Model 3)</b>", tbl_cell), Paragraph("<b>$0 (SAVED)</b>", tbl_cell), Paragraph("<b>₹0 (SAVED)</b>", tbl_cell), Paragraph("<b>Replaces $1,800 physical radar hardware</b>", tbl_cell)],
        [Paragraph("<b>Total Production Cost</b>", tbl_cell), Paragraph("<b>Complete Embedded ADAS Kit</b>", tbl_cell), Paragraph("<b>~$1,450</b>", tbl_cell), Paragraph("<b>~₹1,20,500</b>", tbl_cell), Paragraph("<b>Production-grade Level 3/4 ADAS</b>", tbl_cell)],
    ]
    t_bom = Table(bom_table, colWidths=[125, 160, 75, 75, 312])
    t_bom.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#e2e8f0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#64748b")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_bom)
    story.append(Spacer(1, 8))

    story.append(Paragraph("• <b>Can it be mounted on a Scooty / Bike? ➔ NO (Commercially Infeasible):</b>", ppt_bullet))
    story.append(Paragraph("1. <i>Economic Inversion:</i> ₹1.2 Lakh ADAS kit on an ₹85,000 scooty is economically non-viable (1.4x the vehicle's total cost).<br/>"
                           "2. <i>Motorcycle Leaning Dynamics:</i> A 20°–40° lean angle during turns disrupts fixed FOV LiDAR ground calibration.<br/>"
                           "3. <i>Battery Drain:</i> 30W electrical load will drain a small 4Ah scooter battery in under 1.5 hours.<br/>"
                           "4. <i>Target Deployment Platforms:</i> <b>Commercial Delivery Vans (Tata Ace), Electric Auto-Rickshaws, and Passenger Cars</b>.", ppt_bullet))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 16: CONCLUSION & FUTURE WORK
    # ==============================================================================
    story.append(Paragraph("Conclusion, Limitations &amp; Future Roadmap", slide_title_style))
    story.append(Paragraph("Research Summary, Engineering Assumptions &amp; Deployment Next Steps", slide_subtitle_style))

    story.append(Paragraph("• <b>Research Contributions Summary:</b>", ppt_bullet))
    story.append(Paragraph("1. Built an integrated tri-modal perception, physical radar synthesis, and predictive planning pipeline for unstructured Indian traffic.<br/>"
                           "2. Achieved 1.42 pixel camera-LiDAR reprojection accuracy and 1.12m ADE trajectory prediction error.<br/>"
                           "3. Demonstrated 99.33% collision avoidance reliability with 76.4% steering jitter reduction in 3-zone Frenet frame.<br/>"
                           "4. Implemented a zero-dependency HTML5 Canvas glass cockpit dashboard running offline at 60 FPS.", ppt_bullet))

    story.append(Paragraph("• <b>Engineering Assumptions &amp; Limitations:</b>", ppt_bullet))
    story.append(Paragraph("1. Ground plane is assumed locally planar within 55m radius; extreme road banking requires 3D mesh modeling.<br/>"
                           "2. Radar multipath ground bounce is modeled via Gaussian stochastic approximation rather than full EM ray-tracing.", ppt_bullet))

    story.append(Paragraph("• <b>Future Research Roadmap:</b>", ppt_bullet))
    story.append(Paragraph("• <b>Phase 1:</b> Integration of High-Definition Vector Maps and Lane Graph Neural Networks (LaneGCN) for contextual topological guidance.<br/>"
                           "• <b>Phase 2:</b> Closed-loop Hardware-in-the-Loop (HIL) deployment on a Drive-by-Wire electric vehicle testbed.", ppt_bullet))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 17: QUESTIONS & DISCUSSION
    # ==============================================================================
    story.append(Spacer(1, 45))
    story.append(Paragraph("QUESTIONS &amp; DISCUSSION", ParagraphStyle('QTitle', fontName='Helvetica-Bold', fontSize=26, leading=32, textColor=colors.HexColor("#000000"), alignment=1)))
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#0f172a"), spaceAfter=16, spaceBefore=6))
    story.append(Paragraph("Autonomous Mobility System: Perception, Prediction &amp; Path Planning for Unstructured Indian Road Conditions", ParagraphStyle('QSub', fontName='Helvetica-Bold', fontSize=13, leading=18, textColor=colors.HexColor("#334155"), alignment=1)))
    story.append(Spacer(1, 20))

    recap_data = [
        [Paragraph("<b>Key Defense Takeaways</b>", tbl_hdr)],
        [Paragraph("• Real-world unstructured Indian Driving Dataset (IDD) validation across Urban, Arterial &amp; Highway domains.<br/>"
                   "• <b>$0 Radar Hardware Cost</b> achieved via closed-form 77 GHz FMCW physics synthesis (TI AWR1843 equivalent).<br/>"
                   "• <b>99.33% Collision Avoidance Reliability</b> via 3-Zone Frenet optimal quintic polynomial path planning.<br/>"
                   "• <b>Interactive Glass Cockpit HUD</b> operating at 60 FPS locally without external cloud or server dependencies.<br/>"
                   "• <b>Solid-State LiDAR BOM Analysis</b> proving economic viability for Commercial Fleets, EVs &amp; Passenger Cars.", tbl_cell)]
    ]
    t_recap = Table(recap_data, colWidths=[650])
    t_recap.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#64748b")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_recap)

    story.append(Spacer(1, 25))
    story.append(Paragraph("<b>Author:</b> Hari Prasad  |  <b>Project:</b> PPD-II Autonomous Vehicles  |  <b>GitHub:</b> https://github.com/Hari92004/PPD-II.git", ParagraphStyle('QMeta', fontName='Helvetica', fontSize=9, textColor=colors.HexColor("#64748b"), alignment=1)))

    # Build
    print("[*] Generating Black & White PPT Slide Deck PDF...")
    doc.build(story, canvasmaker=PPTSlideCanvas)
    print(f"[+] PPT Deck Generation Complete! Saved to: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    generate_ppt_deck()
