"""
Validate YOLO26 Model on Real Indian Road Scene Image
Loads model1_camera_yolo.onnx, runs inference, draws bounding boxes and confidence tags,
and exports the annotated result to model1_yolo26_validation_result.png.
"""

import os
import time
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

CLASSES = ['auto_rickshaw', 'cow_cattle', 'pedestrian', 'two_wheeler', 'pothole', 'truck_bus', 'car']
COLORS = [
    (255, 215, 0),   # auto_rickshaw: Gold
    (205, 133, 63),  # cow_cattle: Sandy Brown
    (0, 255, 127),   # pedestrian: Spring Green
    (30, 144, 255),  # two_wheeler: Dodger Blue
    (220, 20, 60),   # pothole: Crimson
    (255, 69, 0),    # truck_bus: Orange Red
    (147, 112, 219)  # car: Medium Purple
]

def run_validation():
    base_dir = r"D:\PPD2"
    img_path = os.path.join(base_dir, "Dataset", "IDD_Detection", "images", "frontFar", "BLR-2018-03-22_17-39-26_2_frontFar", "0000060.jpg")
    onnx_path = os.path.join(base_dir, "Model1", "model1_camera_yolo.onnx")
    output_path = os.path.join(base_dir, "model1_yolo26_validation_result.png")

    print("=" * 75)
    print("      MODEL 1: YOLO26 SINGLE IMAGE VALIDATION TEST")
    print("=" * 75)
    print(f"Test Image  : {img_path}")
    print(f"ONNX Model  : {onnx_path}")

    pil_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    print(f"Resolution  : {orig_w} x {orig_h}")

    # Resize to 640x640 for YOLO26
    img_resized = pil_img.resize((640, 640), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    inp = img_arr.transpose(2, 0, 1)[np.newaxis, ...]

    # Load ONNX Session
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inp_name = sess.get_inputs()[0].name

    # Inference Benchmark
    t0 = time.time()
    preds = sess.run(None, {inp_name: inp})[0]
    latency_ms = (time.time() - t0) * 1000

    draw = ImageDraw.Draw(pil_img)
    detections = []

    sx = orig_w / 640.0
    sy = orig_h / 640.0

    # Parse predictions: [1, 300, 6] -> [x1, y1, x2, y2, conf, cls_id]
    for box in preds[0]:
        conf = float(box[4])
        if conf > 0.20:
            cls_id = int(box[5])
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
            x1, y1 = float(box[0]) * sx, float(box[1]) * sy
            x2, y2 = float(box[2]) * sx, float(box[3]) * sy
            w, h = x2 - x1, y2 - y1
            color = COLORS[cls_id % len(COLORS)]

            # Draw 3D-like thick bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            label_text = f"{cls_name.upper()} {conf*100:.1f}%"

            # Tag banner
            tag_h = 24
            tag_w = len(label_text) * 10 + 10
            draw.rectangle([x1, max(0, y1 - tag_h), x1 + tag_w, y1], fill=color)
            draw.text((x1 + 6, max(0, y1 - tag_h + 4)), label_text, fill=(0, 0, 0))

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)]
            })

    pil_img.save(output_path, quality=95)

    print("\n" + "=" * 75)
    print(f"Inference Latency : {latency_ms:.2f} ms ({1000.0/latency_ms:.1f} FPS)")
    print(f"Total Detections  : {len(detections)} obstacles detected")
    print("-" * 75)
    print(f"{'Class Name':<18} {'Confidence':<15} {'Bounding Box [X, Y, W, H]'}")
    print("-" * 75)
    for d in detections:
        print(f"{d['class']:<18} {d['confidence']*100:>6.1f}%         {d['bbox']}")

    print("-" * 75)
    print(f"Annotated Image Saved: {output_path}")
    print("=" * 75)

if __name__ == "__main__":
    run_validation()

