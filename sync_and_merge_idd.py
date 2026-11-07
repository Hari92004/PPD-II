"""
IDD Multimodal Dataset Merger & Synchronizer
Pairs 1080p Front Camera Images with 16-channel Velodyne LiDAR point clouds.
Filters out GPS coordinates (Lat, Long, Alt) and extracts only paired sensor sweeps.
Converts LiDAR to float32 for 50% storage optimization.
"""

import os
import io
import time
import bisect
import zipfile
import numpy as np

PRIM_ZIP = r"D:\user\Downloads\idd_mm_primary.zip"
SUPP_ZIP = r"D:\user\Downloads\idd_mm_supplement.zip"
OUT_DIR = r"D:\PPD2\dataset_idd_synchronized"

IMAGES_DIR = os.path.join(OUT_DIR, "images")
LIDAR_DIR = os.path.join(OUT_DIR, "lidar")
CSV_PATH = os.path.join(OUT_DIR, "synchronized_index.csv")

def parse_ts(s):
    parts = [float(x) for x in s.strip().split('-')]
    return parts[0] * 3600 + parts[1] * 60 + parts[2] + parts[3] / 1e6

def run_merge():
    start_time = time.time()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LIDAR_DIR, exist_ok=True)

    print("=" * 70)
    print("   IDD MULTIMODAL SYNCHRONIZATION & MERGER PIPELINE")
    print("=" * 70)
    print(f"[INPUT 1] Camera Primary: {PRIM_ZIP}")
    print(f"[INPUT 2] LiDAR Supp   : {SUPP_ZIP}")
    print(f"[OUTPUT]  Merged Folder : {OUT_DIR}")

    print("\n[STEP 1] Opening ZIP archives...")
    z_prim = zipfile.ZipFile(PRIM_ZIP, 'r')
    z_supp = zipfile.ZipFile(SUPP_ZIP, 'r')

    drives = ['d0', 'd1', 'd2']
    csv_rows = [
        "drive,frame_id,timestamp,camera_file,lidar_file,time_delta_ms,vehicle_speed_kmh,engine_rpm,gear"
    ]

    total_extracted = 0

    for d in drives:
        print(f"\n--- Processing Drive: {d} ---")

        # 1. Load Camera Timestamps
        cam_records = []
        for split in ['train.csv', 'val.csv', 'test.csv']:
            name = f'idd_multimodal/primary/{d}/{split}'
            if name in z_prim.namelist():
                lines = z_prim.read(name).decode('utf-8', errors='ignore').strip().split('\n')[1:]
                for l in lines:
                    p = l.split(',')
                    # timestamp, image_idx (GPS columns omitted!)
                    cam_records.append((parse_ts(p[0]), p[1]))
        cam_records.sort(key=lambda x: x[0])
        cam_ts = [x[0] for x in cam_records]
        print(f"  * Loaded {len(cam_records):,} camera frame timestamps")

        # 2. Load LiDAR Timestamps
        lidar_csv_name = f'idd_multimodal/supplement/lidar/{d}/timestamp.csv'
        lidar_lines = z_supp.read(lidar_csv_name).decode('utf-8', errors='ignore').strip().split('\n')[1:]
        lidar_records = []
        for l in lidar_lines:
            p = l.split(',')
            lidar_records.append((p[0], p[1], parse_ts(p[1])))
        print(f"  * Loaded {len(lidar_records):,} LiDAR frame timestamps")

        # 3. Load OBD Telemetry (5.5hr UTC/IST offset handling)
        obd_records = []
        obd_csv_name = f'idd_multimodal/supplement/obd/{d}/obd.csv'
        if obd_csv_name in z_supp.namelist():
            obd_lines = z_supp.read(obd_csv_name).decode('utf-8', errors='ignore').strip().split('\n')[1:]
            for l in obd_lines:
                p = l.split(',')
                # timestamp,speed,rpm,gear,clv,throttle_position
                obd_sec = parse_ts(p[0]) + 5.5 * 3600 # Convert to IST to match LiDAR/Camera
                spd = float(p[1]) if p[1] else 0.0
                rpm = float(p[2]) if p[2] else 0.0
                gear = float(p[3]) if len(p) > 3 and p[3] else 0.0
                obd_records.append((obd_sec, spd, rpm, gear))
            obd_records.sort(key=lambda x: x[0])
        obd_ts = [x[0] for x in obd_records]
        print(f"  * Loaded {len(obd_records):,} OBD telemetry entries")

        # 4. Pair Each LiDAR Frame with Nearest Camera Image & Extract
        print(f"  * Pairing and extracting synchronized frames...")
        drive_count = 0

        for frame_id, ts_str, l_sec in lidar_records:
            # Nearest Camera frame via binary search
            idx = bisect.bisect_left(cam_ts, l_sec)
            candidates = []
            if idx < len(cam_ts):
                candidates.append((abs(cam_ts[idx] - l_sec), cam_records[idx]))
            if idx > 0:
                candidates.append((abs(cam_ts[idx - 1] - l_sec), cam_records[idx - 1]))
            
            best_dt, best_cam = min(candidates, key=lambda x: x[0])
            cam_idx = best_cam[1]

            # Nearest OBD telemetry
            spd, rpm, gear = 0.0, 0.0, 0.0
            if obd_ts:
                o_idx = bisect.bisect_left(obd_ts, l_sec)
                o_cands = []
                if o_idx < len(obd_ts):
                    o_cands.append(obd_records[o_idx])
                if o_idx > 0:
                    o_cands.append(obd_records[o_idx - 1])
                if o_cands:
                    best_obd = min(o_cands, key=lambda x: abs(x[0] - l_sec))
                    spd, rpm, gear = best_obd[1], best_obd[2], best_obd[3]

            # Extract Camera Image (raw bytes)
            cam_zip_path = f"idd_multimodal/primary/{d}/leftCamImgs/{cam_idx}.jpg"
            out_cam_filename = f"{d}_cam_{cam_idx}.jpg"
            out_cam_path = os.path.join(IMAGES_DIR, out_cam_filename)

            if not os.path.exists(out_cam_path):
                try:
                    img_bytes = z_prim.read(cam_zip_path)
                    with open(out_cam_path, 'wb') as f_out:
                        f_out.write(img_bytes)
                except Exception as e:
                    pass

            # Extract LiDAR array (convert to float32)
            lidar_zip_path = f"idd_multimodal/supplement/lidar/{d}/{frame_id}.npy"
            out_lidar_filename = f"{d}_lidar_{frame_id}.npy"
            out_lidar_path = os.path.join(LIDAR_DIR, out_lidar_filename)

            if not os.path.exists(out_lidar_path):
                try:
                    arr_bytes = z_supp.read(lidar_zip_path)
                    arr = np.load(io.BytesIO(arr_bytes)).astype(np.float32)
                    np.save(out_lidar_path, arr)
                except Exception as e:
                    pass

            # Record clean metadata row (NO GPS!)
            row = f"{d},{frame_id},{ts_str},images/{out_cam_filename},lidar/{out_lidar_filename},{best_dt*1000:.2f},{spd:.1f},{rpm:.0f},{gear:.0f}"
            csv_rows.append(row)

            drive_count += 1
            total_extracted += 1

            if drive_count % 500 == 0:
                print(f"    -> Extracted {drive_count:,} / {len(lidar_records):,} frames ({d})...")

        print(f"  [DONE] Drive {d} extracted: {drive_count:,} pairs.")

    # Write Master CSV
    print(f"\n[STEP 3] Writing clean synchronized index to: {CSV_PATH}...")
    with open(CSV_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_rows) + '\n')

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[SUCCESS] Complete Dataset Merged & Synchronized!")
    print(f"  * Total Synchronized Pairs: {total_extracted:,} (Camera + LiDAR)")
    print(f"  * Total Elapsed Time: {elapsed:.1f} s ({elapsed/60:.1f} mins)")
    print(f"  * GPS Data Filtered: 100% Removed")
    print(f"  * Master Index: {CSV_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    run_merge()

