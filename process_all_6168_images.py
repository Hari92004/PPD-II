"""
Processes all 6,168 extracted images from dataset_idd_synchronized with Model 1 (YOLOv8) only.
Streams directly into 1080p MP4 videos at native 10 FPS:
  1. model1_drive0_urban_full.mp4 (2,074 frames, ~3.45 min)
  2. model1_drive1_arterial_full.mp4 (2,007 frames, ~3.34 min)
  3. model1_drive2_highway_full.mp4 (2,087 frames, ~3.48 min)
  4. model1_all_6168_complete.mp4 (All 6,168 frames, ~10.28 min)
"""

import os
import glob
import time
import cv2
import pandas as pd
from ultralytics import YOLO

BASE_DIR = r"D:\PPD2"
IDD_DIR = os.path.join(BASE_DIR, "dataset_idd_synchronized")
IMG_DIR = os.path.join(IDD_DIR, "images")
INDEX_CSV = os.path.join(IDD_DIR, "synchronized_index.csv")

OUT_DIR = os.path.join(BASE_DIR, "model1_full_dataset_videos")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR_GREEN = (0, 230, 90)
COLOR_TEXT_BG = (0, 200, 75)
COLOR_TEXT = (0, 0, 0)
FPS = 10.0

def process_and_draw(frame, model):
    results = model.predict(frame, device=0, imgsz=640, conf=0.25, verbose=False)
    for r in results:
        for box in r.boxes:
            cname = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cname in ['car', 'motorcycle', 'bus', 'truck', 'person', 'bicycle']:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                mapped_cls = 'TRUCK_BUS' if cname in ['bus', 'truck'] else ('TWO_WHEELER' if cname in ['motorcycle', 'bicycle'] else cname.upper())

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2, cv2.LINE_AA)
                lbl = f"{mapped_cls} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), COLOR_TEXT_BG, -1)
                cv2.putText(frame, lbl, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    return frame

def main():
    print("=" * 75)
    print("  BATCH PROCESSING ALL 6,168 IMAGES WITH MODEL 1 (YOLOV8) ONLY")
    print("=" * 75)

    # 1. Load Synchronized Index to preserve true sequential order
    df = pd.read_csv(INDEX_CSV)
    # Filter only rows whose camera_file exists on disk
    valid_rows = []
    for _, row in df.iterrows():
        p = os.path.join(IDD_DIR, row['camera_file'])
        if os.path.exists(p):
            valid_rows.append(row)
    df_valid = pd.DataFrame(valid_rows)
    total_valid = len(df_valid)
    print(f"Verified {total_valid} existing camera image files in chronological sequence.")

    # Model 1
    print("Loading Model 1 (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    # Master Video Writer
    master_path = os.path.join(OUT_DIR, "model1_all_6168_complete.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    master_writer = cv2.VideoWriter(master_path, fourcc, FPS, (1920, 1080))

    drives = ['d0', 'd1', 'd2']
    drive_names = {
        'd0': 'Drive 0: MG Road Urban Sector',
        'd1': 'Drive 1: Outer Ring Arterial Corridor',
        'd2': 'Drive 2: Suburban Highway Bypass'
    }

    t_global_start = time.time()
    global_idx = 0

    for d in drives:
        d_df = df_valid[df_valid['drive'] == d]
        d_count = len(d_df)
        print(f"\n--- Starting {drive_names[d]} ({d_count} frames) ---")

        drive_path = os.path.join(OUT_DIR, f"model1_{d}_full.mp4")
        drive_writer = cv2.VideoWriter(drive_path, fourcc, FPS, (1920, 1080))

        t_drive_start = time.time()
        for i, (_, row) in enumerate(d_df.iterrows()):
            fpath = os.path.join(IDD_DIR, row['camera_file'])
            frame = cv2.imread(fpath)
            if frame is None:
                continue

            processed_frame = process_and_draw(frame, model)

            drive_writer.write(processed_frame)
            master_writer.write(processed_frame)

            global_idx += 1
            if (i + 1) % 100 == 0 or i == d_count - 1:
                el_drive = time.time() - t_drive_start
                el_total = time.time() - t_global_start
                fps = (i + 1) / (el_drive + 1e-5)
                eta_min = (total_valid - global_idx) / (fps + 1e-5) / 60
                pct = (global_idx / total_valid) * 100
                print(f"  [{d.upper()}] {i+1:04d}/{d_count:04d} frames | Total: {global_idx:04d}/{total_valid:04d} ({pct:.1f}%) | Speed: {fps:.1f} FPS | ETA: {eta_min:.1f} min")

        drive_writer.release()
        d_mb = os.path.getsize(drive_path) / (1024 * 1024)
        print(f"[SAVED] {drive_path} ({d_mb:.1f} MB, {d_count/FPS:.1f}s)")

    master_writer.release()
    m_mb = os.path.getsize(master_path) / (1024 * 1024)
    tot_time = time.time() - t_global_start
    print(f"\n" + "=" * 75)
    print(f"[ALL COMPLETE] Processed all {total_valid} images with Model 1 in {tot_time/60:.1f} minutes.")
    print(f"Master Video: {master_path} ({m_mb:.1f} MB, {total_valid/FPS:.1f}s = {total_valid/FPS/60:.2f} min)")
    print(f"Output Directory: {OUT_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
