"""
Validation Script for Model 2: 3D LiDAR Object Detector (PointPillars)
Tests model2_lidar.onnx and best_model2_lidar.pth on sample_pointcloud.npy.
Generates Bird's-Eye-View (BEV) visualization with predicted 3D bounding boxes.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import onnxruntime as ort

def validate_lidar(
    onnx_path=r"D:\PPD2\model2\model2_lidar.onnx",
    sample_path=r"D:\PPD2\model2\sample_pointcloud.npy",
    output_image=r"D:\PPD2\model2_lidar_validation_result.png",
    conf_threshold=0.35
):
    print("=" * 65)
    print("   MODEL 2 (3D LiDAR PointPillars) VALIDATION INFERENCE TEST")
    print("=" * 65)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX Model not found at: {onnx_path}")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample point cloud not found at: {sample_path}")

    # Load point cloud
    pcd = np.load(sample_path).astype(np.float32)
    print(f"[DATA] Loaded Point Cloud: {pcd.shape} points (X, Y, Z, Intensity)")
    print(f"       X range: [{pcd[:,0].min():.2f}, {pcd[:,0].max():.2f}] m")
    print(f"       Y range: [{pcd[:,1].min():.2f}, {pcd[:,1].max():.2f}] m")
    print(f"       Z range: [{pcd[:,2].min():.2f}, {pcd[:,2].max():.2f}] m")

    # Prepare batch tensor: [1, 16384, 4]
    input_tensor = np.expand_dims(pcd[:16384, :4], axis=0)

    # Initialize ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Benchmark Latency (10 runs)
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        raw_output = sess.run([output_name], {input_name: input_tensor})[0]
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = float(np.mean(latencies[1:])) # ignore warmup
    print(f"\n[EXECUTION] Provider: {sess.get_providers()[0]}")
    print(f"[BENCHMARK] Average Inference Latency: {avg_latency:.2f} ms")

    # Parse Detections
    classes = ['Vehicle', 'Pedestrian', 'Cyclist']
    class_colors = {'Vehicle': '#00FFAA', 'Pedestrian': '#FF4444', 'Cyclist': '#FFAA00'}

    predictions = raw_output[0] # [20, 12]
    detected_objects = []

    print(f"\n[DETECTIONS] Parsed 3D Obstacles (Conf > {conf_threshold}):")
    print(f"{'Idx':<4} {'Class':<12} {'Conf':<8} {'X (m)':<8} {'Y (m)':<8} {'Z (m)':<8} {'W (m)':<8} {'L (m)':<8} {'H (m)':<8} {'Yaw (deg)':<10}")
    print("-" * 88)

    for i, box in enumerate(predictions):
        conf = float(box[11])
        if conf >= conf_threshold:
            cls_logits = box[8:11]
            cls_id = int(np.argmax(cls_logits))
            cls_name = classes[cls_id] if cls_id < len(classes) else "Unknown"
            
            x, y, z = float(box[0]), float(box[1]), float(box[2])
            w, l, h = float(box[3]), float(box[4]), float(box[5])
            sin_yaw, cos_yaw = float(box[6]), float(box[7])
            yaw_rad = np.arctan2(sin_yaw, cos_yaw)
            yaw_deg = np.degrees(yaw_rad)

            detected_objects.append({
                'class': cls_name,
                'conf': conf,
                'x': x, 'y': y, 'z': z,
                'w': w, 'l': l, 'h': h,
                'yaw_rad': yaw_rad,
                'yaw_deg': yaw_deg
            })

            print(f"{len(detected_objects):<4} {cls_name:<12} {conf:<8.3f} {x:<8.2f} {y:<8.2f} {z:<8.2f} {w:<8.2f} {l:<8.2f} {h:<8.2f} {yaw_deg:<10.1f}")

    if not detected_objects:
        print("  (No objects exceeded threshold, showing highest scoring proposal)")
        best_idx = int(np.argmax(predictions[:, 11]))
        box = predictions[best_idx]
        cls_name = classes[int(np.argmax(box[8:11]))]
        print(f"  Top proposal: {cls_name} | Conf: {box[11]:.3f} at [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}]")

    # Generate Bird's Eye View (BEV) Plot
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')

    # Scatter LiDAR points (BEV: X=Lateral, Y=Longitudinal)
    subsample = pcd[::2] # speed up plotting
    intensities = subsample[:, 3]
    scatter = ax.scatter(
        subsample[:, 1], # Y lateral
        subsample[:, 0], # X longitudinal
        c=intensities,
        cmap='viridis',
        s=0.6,
        alpha=0.6
    )

    # Ego vehicle center marker
    ax.plot(0, 0, marker='^', markersize=14, color='#00FFFF', label='Ego Vehicle')

    # Draw 3D Bounding Boxes in BEV (Top-down)
    for obj in detected_objects:
        # In nuScenes coordinates: X=Lateral, Y=Longitudinal or vice versa
        # In BEV, center = (obj['y'], obj['x']), dimensions = obj['w'], obj['l']
        color = class_colors.get(obj['class'], '#FFFFFF')
        w, l = obj['w'], obj['l']
        cx, cy = obj['y'], obj['x']
        yaw = obj['yaw_deg']

        rect = patches.Rectangle(
            (cx - w/2, cy - l/2),
            w, l,
            angle=yaw,
            rotation_point='center',
            linewidth=2.0,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            cx, cy + l/2 + 0.8,
            f"{obj['class']}: {obj['conf']:.2f}",
            color=color,
            fontsize=9,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#161B22', edgecolor=color, alpha=0.8)
        )

    # Formatting
    ax.set_xlim(-40, 40)
    ax.set_ylim(-10, 50)
    ax.set_xlabel("Lateral Distance Y (m)", color='white', fontsize=11)
    ax.set_ylabel("Longitudinal Distance X (m)", color='white', fontsize=11)
    ax.set_title(f"Model 2: 3D LiDAR Object Detection (PointPillars BEV)\nInference Latency: {avg_latency:.2f} ms | Detected Obstacles: {len(detected_objects)}",
                 color='white', fontsize=12, pad=12)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', color='#21262D', alpha=0.7)
    
    # Range rings (10m, 20m, 30m, 40m)
    for r in [10, 20, 30, 40]:
        circle = plt.Circle((0, 0), r, fill=False, color='#30363D', linestyle=':', linewidth=1)
        ax.add_patch(circle)
        ax.text(0.5, r, f"{r}m", color='#8B949E', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_image, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"\n[SAVED] BEV Visualization saved: {output_image}")
    return avg_latency, detected_objects

if __name__ == "__main__":
    validate_lidar()

