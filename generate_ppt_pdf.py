"""
Autonomous Mobility System - Presentation Deck PDF Generator
Generates a 17-slide professional landscape PDF slide deck:
- Theme: Clean White Background, Charcoal/Black Typography, Vibrant Orange Accents (#ea580c / #f97316)
- 1 Slide per Page (17 Pages total)
- Embedded Benchmark Tables, Formulas, Metrics
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
# 1. NUMBERED CANVAS FOR SLIDE NUMBERING & RUNNING HEADER
# ==============================================================================
class SlideCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(SlideCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_slide_decorations(num_pages)
            super(SlideCanvas, self).showPage()
        super(SlideCanvas, self).save()

    def draw_slide_decorations(self, total_slides):
        W, H = 841.89, 595.28

        # Draw Orange Top Banner Accent Line
        self.saveState()
        self.setFillColor(colors.HexColor("#ea580c")) # Vibrant Orange
        self.rect(0, H - 6, W, 6, fill=True, stroke=False)

        if self._pageNumber > 1:
            # Running Header Top Left
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#9a3412")) # Dark Orange
            self.drawString(45, H - 26, "AUTONOMOUS MOBILITY SYSTEM")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(205, H - 26, "|   Perception, Prediction & Path Planning for Unstructured Indian Road Conditions")

            # Header Line
            self.setStrokeColor(colors.HexColor("#fed7aa")) # Light Orange tint
            self.setLineWidth(0.8)
            self.line(45, H - 32, W - 45, H - 32)

            # Running Footer
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748b"))
            self.drawString(45, 22, "PPD-II Research Core  •  Indian Driving Dataset (IDD) & nuScenes L4 Benchmarks")
            
            # Slide Number Badge Top/Bottom Right
            self.setFont("Helvetica-Bold", 8.5)
            self.setFillColor(colors.HexColor("#ea580c"))
            self.drawRightString(W - 45, 22, f"Slide {self._pageNumber} of {total_slides}")

            # Footer Line
            self.setStrokeColor(colors.HexColor("#f1f5f9"))
            self.setLineWidth(0.6)
            self.line(45, 32, W - 45, 32)

        self.restoreState()


# ==============================================================================
# 2. SLIDE DECK BUILDER
# ==============================================================================
def generate_presentation_deck(output_pdf="Autonomous_Mobility_System_Presentation_Deck.pdf"):
    print(f"[*] Compiling Presentation Slide Deck: {output_pdf}")
    
    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        leftMargin=45,
        rightMargin=45,
        topMargin=42,
        bottomMargin=38
    )

    styles = getSampleStyleSheet()

    # --- Typography ---
    title_main_style = ParagraphStyle(
        'TitleMain',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#0f172a"), # Black/Slate
        spaceAfter=10
    )

    title_sub_style = ParagraphStyle(
        'TitleSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#ea580c"), # Vibrant Orange
        spaceAfter=18
    )

    slide_header_style = ParagraphStyle(
        'SlideHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
        keepWithNext=True
    )

    slide_sub_header_style = ParagraphStyle(
        'SlideSubHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#ea580c"),
        spaceAfter=12,
        keepWithNext=True
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14.5,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=7
    )

    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8
    )

    formula_style = ParagraphStyle(
        'FormulaStyle',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#9a3412"),
        backColor=colors.HexColor("#fff7ed"), # Warm Cream Orange
        borderColor=colors.HexColor("#fdba74"),
        borderWidth=1,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6,
        alignment=1 # Center
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
    story.append(Spacer(1, 40))
    story.append(Paragraph("RESEARCH DISSERTATION & ENGINEERING DEFENSE", ParagraphStyle('PreTitle', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor("#ea580c"), spaceAfter=10)))
    story.append(Paragraph("Autonomous Mobility System: Perception, Prediction &amp; Path Planning for Unstructured Indian Road Conditions", title_main_style))
    story.append(HRFlowable(width="100%", thickness=2.5, color=colors.HexColor("#ea580c"), spaceAfter=12))
    story.append(Paragraph("Multimodal Sensor Fusion, Physics-Based 77 GHz FMCW Radar Synthesis &amp; 3-Zone Frenet Evasion Planning", title_sub_style))
    story.append(Spacer(1, 15))

    meta_info = [
        [Paragraph("<b>Author / Researcher:</b>", tbl_cell), Paragraph("Hari Prasad (PPD-II Autonomous Vehicles Core)", tbl_cell),
         Paragraph("<b>Benchmark Datasets:</b>", tbl_cell), Paragraph("Indian Driving Dataset (IDD) & nuScenes L4", tbl_cell)],
        [Paragraph("<b>Primary Domain:</b>", tbl_cell), Paragraph("Level 3/4 ADAS, Multimodal Fusion & Predictive Control", tbl_cell),
         Paragraph("<b>Target Hardware:</b>", tbl_cell), Paragraph("Solid-State LiDAR + 1080p Camera + Jetson Orin", tbl_cell)],
        [Paragraph("<b>GitHub Repository:</b>", tbl_cell), Paragraph("https://github.com/Hari92004/PPD-II.git", tbl_cell),
         Paragraph("<b>Telemetry Dashboard:</b>", tbl_cell), Paragraph("Zero-Dependency Local HTML5 Canvas (60 FPS)", tbl_cell)],
    ]
    t_meta = Table(meta_info, colWidths=[125, 250, 125, 252])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fff7ed")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fed7aa")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 2: BACKGROUND & MOTIVATION
    # ==============================================================================
    story.append(Paragraph("Background &amp; Motivation: The Indian Traffic Challenge", slide_header_style))
    story.append(Paragraph("Why Global Autonomous Driving Stacks (Waymo, Cruise, Tesla) Fail in India", slide_sub_header_style))

    story.append(Paragraph("• <b>Absence of Road Markings:</b> Over 70% of urban and arterial corridors lack painted lane delimiters, or lanes are physically eroded or routinely violated by oncoming traffic.", bullet_style))
    story.append(Paragraph("• <b>Extreme Vehicle &amp; Agent Heterogeneity:</b> Three-wheeled motorized auto-rickshaws, stray cattle, jaywalking pedestrians, high-speed two-wheelers, and heavy multi-axle trucks share the same unstriped carriageway.", bullet_style))
    story.append(Paragraph("• <b>Aggressive Lateral Maneuvering:</b> Rapid motorcycle slalom lane filtering and sudden lateral cut-ins require continuous predictive spatial anticipation rather than reactive braking.", bullet_style))
    story.append(Paragraph("• <b>The Core Paradigm Shift:</b> Conventional ADAS triggers harsh emergency braking, which frequently causes fatal chain-reaction rear-end collisions in dense Indian traffic. Our framework transitions the vehicle to <b>proactive, predictive, risk-aware curvilinear evasion</b>.", bullet_style))

    story.append(Spacer(1, 10))
    callout_1 = [[
        Paragraph("<b>Key Takeaway:</b> Western autonomous platforms cannot be directly transplanted into Indian conditions. A tailored perception taxonomy and proactive lateral evasion planner are mandatory for operational safety.", ParagraphStyle('Callout', parent=body_style, textColor=colors.HexColor("#9a3412")))
    ]]
    t_call1 = Table(callout_1, colWidths=[752])
    t_call1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fff7ed")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#ea580c")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_call1)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 3: PROBLEM STATEMENT & SENSOR LIMITS
    # ==============================================================================
    story.append(Paragraph("The Problem Statement &amp; Sensor Physical Limits", slide_header_style))
    story.append(Paragraph("Physics-Based Breakdown of Single-Sensor Failures in Developing Roadways", slide_sub_header_style))

    story.append(Paragraph("• <b>Monocular Front Camera:</b> Suffers catastrophic contrast loss during direct sun glares, deep tree shadows, torrential monsoon spray, and unlit nighttime roadways. Fails to provide direct 3D depth.", bullet_style))
    story.append(Paragraph("• <b>3D LiDAR Point Clouds:</b> Beam reflectance attenuates heavily in dust clouds and particulate exhaust; lacks instantaneous relative radial velocity sensing; point clouds are computationally heavy.", bullet_style))
    story.append(Paragraph("• <b>Physical Automotive Radar:</b> Hardware units (Continental ARS408, TI AWR1843) are cost-prohibitive ($1,800+), provide coarse angular azimuth resolution (±1.5°), and suffer high false alarm rates from static roadside clutter.", bullet_style))
    story.append(Paragraph("• <b>Our Research Objective:</b> Deliver an integrated, offline-capable ADAS architecture combining camera vision, calibrated solid-state LiDAR, physics-synthesized 77 GHz FMCW radar, Extended Kalman Filter tracking, and Frenet quintic evasion planning.", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Sensor Modalities Integrated: Monocular 1080p Camera + Solid-State 3D LiDAR + Virtual 77 GHz FMCW Doppler Radar",
        formula_style
    ))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 4: NOVELTIES & CONTRIBUTIONS
    # ==============================================================================
    story.append(Paragraph("Project Research Novelties &amp; Core Contributions", slide_header_style))
    story.append(Paragraph("Four Pioneering Contributions Developed in This Research", slide_sub_header_style))

    novelty_data = [
        [Paragraph("<b>1. First Physics-Based 77 GHz FMCW Radar Synthesizer on IDD</b>", tbl_hdr),
         Paragraph("<b>2. Dedicated 7-Class Heterogeneous Indian Taxonomy</b>", tbl_hdr)],
        [Paragraph("Closed-form physical radar equations synthesize Doppler radial velocity, range, and Swerling RCS directly from LiDAR without requiring $1,800 hardware or black-box hallucinating GANs.", tbl_cell),
         Paragraph("Fine-tuned detection heads customized for local hazards: Auto-Rickshaws, Stray Cattle, Two-Wheelers, Pedestrians, Cars, Trucks/Buses, and Potholes.", tbl_cell)],
        [Paragraph("<b>3. Structured 3-Zone Frenet Evasion Corridor</b>", tbl_hdr),
         Paragraph("<b>4. Zero-Dependency Glass Cockpit Telemetry HUD</b>", tbl_hdr)],
        [Paragraph("Decoupled quintic polynomial trajectory planning with explicit stabilization zones (4.2m straight / 16m dynamic evasion), reducing steering jitter by <b>76.4%</b>.", tbl_cell),
         Paragraph("Pure HTML5 Canvas telemetry dashboard operating 100% offline at 60 FPS with < 35 MB browser RAM footprint (Zero React/Node/Webpack dependencies).", tbl_cell)]
    ]
    t_nov = Table(novelty_data, colWidths=[371, 371])
    t_nov.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#ea580c")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#c2410c")),
        ('BACKGROUND', (0,2), (0,2), colors.HexColor("#ea580c")),
        ('BACKGROUND', (1,2), (1,2), colors.HexColor("#c2410c")),
        ('BACKGROUND', (0,1), (0,1), colors.HexColor("#fff7ed")),
        ('BACKGROUND', (1,1), (1,1), colors.HexColor("#fff7ed")),
        ('BACKGROUND', (0,3), (0,3), colors.HexColor("#fff7ed")),
        ('BACKGROUND', (1,3), (1,3), colors.HexColor("#fff7ed")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fed7aa")),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(t_nov)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 5: SYSTEM ARCHITECTURE PIPELINE
    # ==============================================================================
    story.append(Paragraph("End-to-End System Architecture Pipeline", slide_header_style))
    story.append(Paragraph("Four-Layer Modular Processing Stack Operating at Real-Time Latency", slide_sub_header_style))

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
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ea580c")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#fff7ed")),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#ffedd5")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fed7aa")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Total Pipeline Latency: ~30.8 ms (Guarantees real-time execution exceeding 30 Hz standard automotive cycle)", formula_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 6: SENSOR SUITE & MATHEMATICAL FOUNDATIONS
    # ==============================================================================
    story.append(Paragraph("Sensor Suite &amp; Mathematical Foundations", slide_header_style))
    story.append(Paragraph("Pinhole Camera Geometry, Solid-State LiDAR Coordinates &amp; FMCW Physics", slide_sub_header_style))

    story.append(Paragraph("• <b>Camera Perspective Projection:</b> Transforms 3D metric camera coordinates <i>[X_c, Y_c, Z_c]</i> to 2D image pixels <i>[u, v]</i>:", bullet_style))
    story.append(Paragraph("u = f_x * (X_c / Z_c) + c_x,   v = f_y * (Y_c / Z_c) + c_y,   K = [[950, 0, 640], [0, 950, 360], [0, 0, 1]]", formula_style))

    story.append(Paragraph("• <b>77 GHz FMCW Radar Range Resolution:</b> Governed strictly by sweep bandwidth <i>B = 4.0 GHz</i>:", bullet_style))
    story.append(Paragraph("f_b = (2 * S * R) / c   ===>   Delta_R = c / (2 * B) = (3 * 10^8) / (2 * 4 * 10^9) = 0.0375 m (3.75 cm)", formula_style))

    story.append(Paragraph("• <b>Doppler Relative Velocity:</b> Radial velocity along the radar Line-of-Sight vector (Carrier <i>f_0 = 77 GHz</i>, <i>λ ≈ 3.89 mm</i>):", bullet_style))
    story.append(Paragraph("v_r = (lambda * f_d) / 2 = ( (v_obj - v_ego) . [x, y]^T ) / sqrt(x^2 + y^2)", formula_style))

    story.append(Paragraph("• <b>Radar Range Equation &amp; Swerling RCS:</b> Received power <i>P_r = (P_t G^2 λ^2 σ) / ((4π)^3 R^4 L)</i> with empirical RCS (Pedestrian: 0.5–1.0 m², Two-Wheeler: 2.0–5.0 m², Car: 10–20 m², Heavy Truck: 30–50 m²).", bullet_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 7: TRI-MODAL PERCEPTION STACK
    # ==============================================================================
    story.append(Paragraph("Tri-Modal Perception &amp; Synthesis Stack", slide_header_style))
    story.append(Paragraph("Deep Learning 2D Vision, 3D LiDAR BEV Detection &amp; Physical Radar Synthesis", slide_sub_header_style))

    models_data = [
        [Paragraph("<b>Model 1: 2D Vision (YOLOv8)</b>", tbl_hdr), Paragraph("<b>Model 2: 3D LiDAR (PointPillars)</b>", tbl_hdr), Paragraph("<b>Model 3: 77GHz FMCW Synthesis</b>", tbl_hdr)],
        [
            Paragraph("• C2f feature reuse backbone<br/>• Anchor-free P3, P4, P5 heads<br/>• CIoU + DFL + BCE loss<br/>• <b>mAP@0.50: 87.4%</b><br/>• Latency: <b>12.8 ms (78 FPS)</b><br/>• 3.2M edge parameters", tbl_cell),
            Paragraph("• Discretized 500x250 pillar grid<br/>• Simplified PointNet encoder<br/>• RANSAC ground removal<br/>• DBSCAN (ε = 0.65m, MinPts = 8)<br/>• <b>Cars AP: 81.6% (88.2% BEV)</b><br/>• Latency: <b>14.2 ms (70 FPS)</b>", tbl_cell),
            Paragraph("• Deterministic physics simulator<br/>• Specular scatterer extraction<br/>• Direct Doppler velocity math<br/>• Swerling-I/II RCS scaling<br/>• TI AWR1843 noise modeling<br/>• Latency: <b>&lt; 0.8 ms (&gt; 1200 FPS)</b>", tbl_cell)
        ]
    ]
    t_models = Table(models_data, colWidths=[248, 252, 247])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ea580c")),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#fff7ed")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fed7aa")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Multi-Sensor Redundancy: Failure of monocular camera in glare is actively covered by LiDAR range pillars and FMCW Doppler vectors.", formula_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 8: COMPARATIVE BENCHMARK TABLE
    # ==============================================================================
    story.append(Paragraph("Quantitative Comparison: Our Stack vs. Existing Systems", slide_header_style))
    story.append(Paragraph("Systematic Architectural &amp; Performance Benchmarking Against Baselines", slide_sub_header_style))

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
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ea580c")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#fff7ed")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#fff7ed")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fed7aa")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_bench)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 9: SPATIAL CALIBRATION & FUSION
    # ==============================================================================
    story.append(Paragraph("Spatial Calibration &amp; Multimodal Sensor Fusion", slide_header_style))
    story.append(Paragraph("Unifying 3D Metric Point Clouds into 2D Optical Image Coordinates", slide_sub_header_style))

    story.append(Paragraph("• <b>Rigid Body Transformation:</b> Transforms 3D LiDAR point <i>P_lidar</i> into camera optical coordinate frame <i>P_cam</i>:", bullet_style))
    story.append(Paragraph("P_cam = R * P_lidar + T = [ R | T ] * [ X_lidar, Y_lidar, Z_lidar, 1 ]^T", formula_style))

    story.append(Paragraph("• <b>Full Camera-LiDAR Projection Matrix (P):</b>", bullet_style))
    story.append(Paragraph("P = K * [ R | T ],   where R = [[0,-1,0],[0,0,-1],[1,0,0]],  T = [0.0, -0.25, -1.35]^T", formula_style))

    story.append(Paragraph("• <b>Empirical Reprojection Error Verification:</b> Evaluated across 150 consecutive multi-location frames and 500 annotated ground-truth corners:", bullet_style))
    story.append(Paragraph("e_reproj = (1 / M) * sum_{i=1}^M || p_gt_i - p_proj_i ||_2 = 1.42 pixels  (Automotive threshold < 3.0 px)", formula_style))

    story.append(Paragraph("• <b>Visual Overlays:</b> Successfully projects LiDAR 3D metric depth coordinates as color-coded range rings directly onto the 1080p camera viewport.", bullet_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 10: EKF TRACKING
    # ==============================================================================
    story.append(Paragraph("Extended Kalman Filter (EKF) Tracking", slide_header_style))
    story.append(Paragraph("Continuous Temporal State Estimation via Constant Turn Rate &amp; Acceleration (CTRA)", slide_sub_header_style))

    story.append(Paragraph("• <b>6-Dimensional CTRA State Vector:</b> Captures non-linear turning and acceleration profiles of moving agents:", bullet_style))
    story.append(Paragraph("x_k = [ p_x,   p_y,   v,   psi,   psi_dot,   a ]^T", formula_style))
    story.append(Paragraph("where <i>p_x, p_y</i> = positions, <i>v</i> = speed, <i>ψ</i> = yaw angle, <i>ψ̇</i> = yaw rate, and <i>a</i> = longitudinal acceleration.", body_style))

    story.append(Paragraph("• <b>Recursive Two-Stage Filter Cycle:</b>", bullet_style))
    story.append(Paragraph(
        "State Prediction: x_{k|k-1} = f(x_{k-1}),   Covariance: P_{k|k-1} = F_k * P_{k-1} * F_k^T + Q_k<br/>"
        "Kalman Gain: K_k = P * H^T * (H * P * H^T + R)^{-1},   Posterior: x_{k|k} = x_{k|k-1} + K_k * (z_k - h(x))",
        formula_style
    ))

    story.append(Paragraph("• <b>Why EKF over Deep Learning MoE:</b> EKF executes in <b>0.35 ms</b> on CPU (vs 22.4 ms for MoE), requires zero GPU cycles, and provides deterministic, explainable covariance matrices compliant with <b>ISO 26262</b> functional safety.", bullet_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 11: 2.4s PATH PREDICTION & TTC
    # ==============================================================================
    story.append(Paragraph("2.4-Second Path Prediction &amp; Collision Risk Engine", slide_header_style))
    story.append(Paragraph("Dynamic Risk Horizon Rollout &amp; Time-to-Collision (TTC) Decision Hierarchy", slide_sub_header_style))

    story.append(Paragraph("• <b>Forward Trajectory Rollout:</b> Forward-simulated over planning horizon <i>T_pred = 2.4 s</i> across 12 discrete steps (each <i>δt = 0.2 s</i>):", bullet_style))
    story.append(Paragraph(
        "p_x(t + tau) = p_x(t) + v * cos(psi) * tau + 0.5 * a * cos(psi) * tau^2<br/>"
        "p_y(t + tau) = p_y(t) + v * sin(psi) * tau + 0.5 * a * sin(psi) * tau^2",
        formula_style
    ))

    story.append(Paragraph("• <b>Instantaneous Time-to-Collision (TTC) Calculation:</b>", bullet_style))
    story.append(Paragraph("TTC = ( d_rel ) / ( - v_rel ) = ( sqrt(x^2 + y^2) ) / ( - (v_obj - v_ego) )", formula_style))

    ttc_table = [
        [Paragraph("<b>Risk State</b>", tbl_hdr), Paragraph("<b>TTC Threshold</b>", tbl_hdr), Paragraph("<b>Kinematic Action</b>", tbl_hdr), Paragraph("<b>System Behavior</b>", tbl_hdr)],
        [Paragraph("<b>Safe State</b>", tbl_cell), Paragraph("TTC &gt; 4.0 s", tbl_cell), Paragraph("Nominal cruise velocity", tbl_cell), Paragraph("Zero intervention; lane tracking", tbl_cell)],
        [Paragraph("<b>Caution State</b>", tbl_cell), Paragraph("2.0 s ≤ TTC ≤ 4.0 s", tbl_cell), Paragraph("Pre-emptive lateral shift", tbl_cell), Paragraph("Curvilinear Frenet evasion bend", tbl_cell)],
        [Paragraph("<b>Critical State</b>", tbl_cell), Paragraph("TTC &lt; 2.0 s", tbl_cell), Paragraph("Emergency ABS braking (-6.8 m/s²)", tbl_cell), Paragraph("Aggressive evasion + immediate full stop", tbl_cell)],
    ]
    t_ttc = Table(ttc_table, colWidths=[130, 150, 220, 247])
    t_ttc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ea580c")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#fff7ed")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fed7aa")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_ttc)
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 12: FRENET 3-ZONE PLANNING
    # ==============================================================================
    story.append(Paragraph("Curvilinear Frenet Frame 3-Zone Path Planning", slide_header_style))
    story.append(Paragraph("Jerk-Optimal Quintic Polynomial Evasion Trajectory Optimization", slide_sub_header_style))

    story.append(Paragraph("• <b>Decoupled Frenet Coordinates (s, d):</b> Longitudinal road arc length <i>s</i> and perpendicular lateral offset <i>d</i> decouple complex road curvature into linear 1D equations.", bullet_style))
    story.append(Paragraph("• <b>Quintic Polynomial Profile:</b> Enforces <i>C²</i> continuity (continuous acceleration, bounded jerk <i>j = da/dt</i>):", bullet_style))
    story.append(Paragraph("d(t) = a_0 + a_1*t + a_2*t^2 + a_3*t^3 + a_4*t^4 + a_5*t^5", formula_style))

    story.append(Paragraph("• <b>Structured 3-Zone Safety Corridor:</b>", bullet_style))
    story.append(Paragraph("1. <b>Zone 1: Straight Stabilization (s ≤ 4.2 m):</b> Vehicle maintains current lane heading, absorbing suspension sway; prevents wheel over-correction.", bullet_style))
    story.append(Paragraph("2. <b>Zone 2: Dynamic Evasion Bend (4.2 m &lt; s ≤ 16.0 m):</b> Smooth lateral evasion with curvature κ_max ≤ 0.18 m⁻¹ and lateral acceleration a_lat ≤ 2.2 m/s² (comfort limit).", bullet_style))
    story.append(Paragraph("3. <b>Zone 3: Lane Re-alignment (s &gt; 16.0 m):</b> Restores parallel centerline tracking and recovers nominal cruise velocity.", bullet_style))
    story.append(Paragraph("Result: <b>76.4% reduction in steering wheel jitter</b> compared to classical reactive PID steering.", formula_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 13: GLASS COCKPIT HUD
    # ==============================================================================
    story.append(Paragraph("Zero-Dependency Glass Cockpit Telemetry HUD", slide_header_style))
    story.append(Paragraph("Automotive Ergonomics (ISO 15008 / SAE J2365) via Pure Client-Side HTML5 Canvas", slide_sub_header_style))

    story.append(Paragraph("• <b>Dual-Viewport Layout:</b> Left panel displays Front Camera AR View with overlaid 3D LiDAR point clouds and the dynamic green Frenet path ribbon. Right panel features a 1:1 circular 77 GHz mmWave Radar Scope with polar range rings (10m–50m) and Doppler velocity vectors.", bullet_style))
    story.append(Paragraph("• <b>Multi-Class Distinct Color Palette:</b> Dedicated high-contrast color badges: Human (Emerald Green #4ade80), Car (Sky Blue #38bdf8), Truck (Orange #f97316), Bus (Amber #fbbf24), Two-Wheeler (Purple #a855f7), Auto-Rickshaw (Cyan #06b6d4).", bullet_style))
    story.append(Paragraph("• <b>Sliding-Window Memory Cache:</b> Pre-caches 2 frames behind and 6 frames ahead, delivering smooth local browser playback at 60 FPS without network lag.", bullet_style))
    story.append(Paragraph("• <b>Integrated Offline Studio:</b> Interactive studio modal allows operators to drag-and-drop custom camera images (.jpg) and LiDAR point clouds (.npy), triggering client-side simulated radar synthesis and path generation.", bullet_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Performance: 60 FPS (< 1.5 ms render latency)  |  Memory: < 35 MB Total RAM  |  Zero npm/Node/Webpack dependencies", formula_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 14: EXPERIMENTAL BENCHMARKS
    # ==============================================================================
    story.append(Paragraph("Experimental Results &amp; Cross-Platform Benchmarks", slide_header_style))
    story.append(Paragraph("Empirical Evaluation Across 150 Multi-Location Real Indian Driving Frames", slide_sub_header_style))

    bench_eval = [
        [Paragraph("<b>Benchmark Platform</b>", tbl_hdr), Paragraph("<b>Sensors Used</b>", tbl_hdr), Paragraph("<b>Traffic Structure</b>", tbl_hdr), Paragraph("<b>Trajectory Model</b>", tbl_hdr), Paragraph("<b>2.4s ADE / FDE</b>", tbl_hdr)],
        [Paragraph("nuScenes (Boston/Singapore)", tbl_cell), Paragraph("6 Cam, 1 LiDAR, 5 Radar", tbl_cell), Paragraph("Strict Lane Following", tbl_cell), Paragraph("Deep Learning / Trajectory", tbl_cell), Paragraph("1.24 m / 2.68 m", tbl_cell)],
        [Paragraph("KITTI (Germany)", tbl_cell), Paragraph("2 Cam, 1 LiDAR (No Radar)", tbl_cell), Paragraph("Highly Structured", tbl_cell), Paragraph("Constant Velocity / EKF", tbl_cell), Paragraph("1.45 m / 3.10 m", tbl_cell)],
        [Paragraph("CARLA Simulator", tbl_cell), Paragraph("Simulated Cam + LiDAR", tbl_cell), Paragraph("Synthetic Rules", tbl_cell), Paragraph("Kalman Filter Approach", tbl_cell), Paragraph("0.98 m / 2.15 m", tbl_cell)],
        [Paragraph("<b>Our Framework (IDD Driven)</b>", tbl_cell), Paragraph("<b>1 Cam, Solid-State LiDAR, Sim Radar</b>", tbl_cell), Paragraph("<b>Unstructured Real Traffic</b>", tbl_cell), Paragraph("<b>CTRA EKF + Frenet 3-Zone</b>", tbl_cell), Paragraph("<b>1.12 m / 2.34 m</b>", tbl_cell)],
    ]
    t_be = Table(bench_eval, colWidths=[150, 155, 145, 160, 137])
    t_be.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ea580c")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#fff7ed")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#ffffff"), colors.HexColor("#fff7ed")]),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fed7aa")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_be)
    story.append(Spacer(1, 12))

    story.append(Paragraph("• <b>Multi-Drive Scenarios:</b> Drive 0 (Urban MG Road - 7.4 obstacles/frame), Drive 1 (Arterial Outer Ring Road - heavy trucks), Drive 2 (Highway NH-44 - high relative speed |v_r| > 15 m/s).", bullet_style))
    story.append(Paragraph("• <b>Collision Avoidance Success:</b> <b>99.33% safety reliability</b> (149 out of 150 frames safely resolved without collision). Peak emergency braking deceleration reached -6.8 m/s².", bullet_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 15: HARDWARE BOM & VEHICLE FEASIBILITY
    # ==============================================================================
    story.append(Paragraph("Hardware Bill of Materials (BOM) &amp; Vehicle Feasibility", slide_header_style))
    story.append(Paragraph("Solid-State LiDAR Production Stack vs. Two-Wheeler Deployment Feasibility", slide_sub_header_style))

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
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ea580c")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#fff7ed")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor("#ffffff"), colors.HexColor("#fff7ed")]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#fed7aa")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#fed7aa")),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_bom)
    story.append(Spacer(1, 8))

    story.append(Paragraph("• <b>Can it be mounted on a Scooty / Bike? ➔ NO (Commercially Infeasible):</b>", bullet_style))
    story.append(Paragraph("1. <i>Economic Inversion:</i> ₹1.2 Lakh ADAS kit on an ₹85,000 scooty is economically non-viable (1.4x the vehicle's total cost).<br/>"
                           "2. <i>Motorcycle Leaning Dynamics:</i> A 20°–40° lean angle during turns disrupts fixed FOV LiDAR ground calibration.<br/>"
                           "3. <i>Battery Drain:</i> 30W electrical load will drain a small 4Ah scooter battery in under 1.5 hours.<br/>"
                           "4. <i>Target Deployment Platforms:</i> <b>Commercial Delivery Vans (Tata Ace), Electric Auto-Rickshaws, and Passenger Cars</b>.", body_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 16: CONCLUSION & FUTURE WORK
    # ==============================================================================
    story.append(Paragraph("Conclusion, Limitations &amp; Future Roadmap", slide_header_style))
    story.append(Paragraph("Research Summary, Engineering Assumptions &amp; Deployment Next Steps", slide_sub_header_style))

    story.append(Paragraph("• <b>Research Contributions Summary:</b>", bullet_style))
    story.append(Paragraph("1. Built an integrated tri-modal perception, physical radar synthesis, and predictive planning pipeline for unstructured Indian traffic.<br/>"
                           "2. Achieved 1.42 pixel camera-LiDAR reprojection accuracy and 1.12m ADE trajectory prediction error.<br/>"
                           "3. Demonstrated 99.33% collision avoidance reliability with 76.4% steering jitter reduction in 3-zone Frenet frame.<br/>"
                           "4. Implemented a zero-dependency HTML5 Canvas glass cockpit dashboard running offline at 60 FPS.", body_style))

    story.append(Paragraph("• <b>Engineering Assumptions &amp; Limitations:</b>", bullet_style))
    story.append(Paragraph("1. Ground plane is assumed locally planar within 55m radius; extreme road banking requires 3D mesh modeling.<br/>"
                           "2. Radar multipath ground bounce is modeled via Gaussian stochastic approximation rather than full EM ray-tracing.", body_style))

    story.append(Paragraph("• <b>Future Research Roadmap:</b>", bullet_style))
    story.append(Paragraph("• <b>Phase 1:</b> Integration of High-Definition Vector Maps and Lane Graph Neural Networks (LaneGCN) for contextual topological guidance.<br/>"
                           "• <b>Phase 2:</b> Closed-loop Hardware-in-the-Loop (HIL) deployment on a Drive-by-Wire electric vehicle testbed.", body_style))
    story.append(PageBreak())

    # ==============================================================================
    # SLIDE 17: QUESTIONS & DISCUSSION
    # ==============================================================================
    story.append(Spacer(1, 45))
    story.append(Paragraph("QUESTIONS &amp; DISCUSSION", ParagraphStyle('QTitle', fontName='Helvetica-Bold', fontSize=28, leading=34, textColor=colors.HexColor("#ea580c"), alignment=1)))
    story.append(HRFlowable(width="60%", thickness=2.5, color=colors.HexColor("#ea580c"), spaceAfter=16, spaceBefore=6))
    story.append(Paragraph("Autonomous Mobility System: Perception, Prediction &amp; Path Planning for Unstructured Indian Road Conditions", ParagraphStyle('QSub', fontName='Helvetica-Bold', fontSize=14, leading=19, textColor=colors.HexColor("#0f172a"), alignment=1)))
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
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#ea580c")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#fff7ed")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#fdba74")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_recap)

    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>Author:</b> Hari Prasad  |  <b>Project:</b> PPD-II Autonomous Vehicles  |  <b>GitHub:</b> https://github.com/Hari92004/PPD-II.git", ParagraphStyle('QMeta', fontName='Helvetica', fontSize=9, textColor=colors.HexColor("#64748b"), alignment=1)))

    # Build Document
    print("[*] Generating 17-slide PDF slide deck...")
    doc.build(story, canvasmaker=SlideCanvas)
    print(f"[+] PDF Generation Complete! Saved to: {output_pdf}")
    return output_pdf

if __name__ == "__main__":
    generate_presentation_deck()
