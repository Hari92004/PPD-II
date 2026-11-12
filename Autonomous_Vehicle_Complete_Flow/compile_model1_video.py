"""
Compiles Model 1 (YOLOv8 2D Camera Vision) Processed Frames into Video.
Physical Camera Sensor Native Rate: 10 FPS (100 ms per sweep)
Output: Full HD 1080p MP4 Video
"""

import os
import glob
import cv2

IMG_DIR = r"D:\PPD2\model1_processed_images"
OUT_VIDEO = r"D:\PPD2\model1_camera_vision_simulation.mp4"
FPS = 10.0  # Native IDD Camera sampling rate (100ms per frame)

def main():
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "model1_frame_*.jpg")))
    total = len(img_files)
    if total == 0:
        print("[ERROR] No processed frames found in", IMG_DIR)
        return

    print("=" * 70)
    print("  COMPILING MODEL 1 CAMERA VISION SIMULATION VIDEO")
    print("=" * 70)
    print(f"Found {total} processed Model 1 frames.")

    first_frame = cv2.imread(img_files[0])
    h, w = first_frame.shape[:2]
    print(f"Frame Resolution: {w}x{h} (Full HD)")
    print(f"Framerate: {FPS} FPS (Real-world synchronized playback)")
    print(f"Expected Duration: {total / FPS:.1f} seconds")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (w, h))

    for i, fpath in enumerate(img_files):
        frame = cv2.imread(fpath)
        if frame is not None:
            writer.write(frame)
        if (i + 1) % 30 == 0 or i == total - 1:
            print(f"  Encoded [{i+1:03d}/{total:03d}] frames...")

    writer.release()
    size_mb = os.path.getsize(OUT_VIDEO) / (1024 * 1024)
    print(f"\n[DONE] Successfully saved video to: {OUT_VIDEO} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
