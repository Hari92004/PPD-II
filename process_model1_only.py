"""
Model 1 (YOLOv8 2D Camera Vision) Dedicated Batch Processor
Strictly processes camera frames with Model 1 only:
  - NO LiDAR points
  - NO 3D bounding boxes
  - NO path planning ribbons
  - NO radar simulation
Outputs pure 2D vision detections with class names and confidence scores.
"""

import os
import time
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO

BASE_DIR = r"D:\PPD2"
IDD_DIR = os.path.join(BASE_DIR, "dataset_idd_synchronized")
INDEX_CSV = os.path.join(IDD_DIR, "synchronized_index.csv")
OUT_DIR = os.path.join(BASE_DIR, "model1_processed_images")

os.makedirs(OUT_DIR, exist_ok=True)

COLOR_GREEN = (0, 230, 90)     # Clean Camera 2D Bounding Box
COLOR_TEXT_BG = (0, 200, 75)
COLOR_TEXT = (0, 0, 0)

def main(num_frames=150):
    print("=" * 75)
    print("  MODEL 1 (YOLOV8) CAMERA VISION BATCH PROCESSING")
    print("=" * 75)

    df_all = pd.read_csv(INDEX_CSV)
    frames_per_drive = num_frames // 3
    chunks = []
    for d in ['d0', 'd1', 'd2']:
        sub = df_all[df_all['drive'] == d]
        start_idx = min(25, max(0, len(sub) - frames_per_drive))
        chunks.append(sub.iloc[start_idx:start_idx + frames_per_drive])
    selected_df = pd.concat(chunks, ignore_index=True)
    total = len(selected_df)

    print(f"Loaded {total} frames across Drive 0 (Urban), Drive 1 (Arterial), Drive 2 (Highway).")
    print("Loading Model 1 (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    t0 = time.time()
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        img_path = os.path.join(IDD_DIR, row['camera_file'])
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        results = model(frame, conf=0.25, verbose=False)

        # Draw ONLY Model 1 2D detections
        for r in results:
            for box in r.boxes:
                cname = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if cname in ['car', 'motorcycle', 'bus', 'truck', 'person', 'bicycle']:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    mapped_cls = 'TRUCK_BUS' if cname in ['bus', 'truck'] else ('TWO_WHEELER' if cname in ['motorcycle', 'bicycle'] else cname.upper())

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2, cv2.LINE_AA)

                    # Draw Label Pill
                    lbl = f"{mapped_cls} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), COLOR_TEXT_BG, -1)
                    cv2.putText(frame, lbl, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

        # Save to output directory
        out_name = f"model1_frame_{idx:04d}_{row['drive']}_{row['frame_id']}.jpg"
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (idx + 1) % 30 == 0 or idx == total - 1:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"  Processed [{idx+1:03d}/{total:03d}] frames ({fps:.1f} FPS)...")

    print(f"\n[DONE] Successfully processed all {total} images with Model 1 only.")
    print(f"[OUTPUT] Saved to: {OUT_DIR}")
    print(f"Total time: {time.time() - t0:.1f} seconds")

if __name__ == "__main__":
    main()

