"""
Generate Camera Vision Only Video (H.264 MP4)
Renders strictly Panel 1: RGB Camera Frame + Calibrated LiDAR Depth Overlay & Tracked Detections.
"""

import os
import sys
import json
import time
import subprocess
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio_ffmpeg

# Import pipeline functions
sys.path.insert(0, r"D:\PPD2")
from pipeline_perception_prediction_planning import (
    BASE_DIR, META_DIR, TemporalEgoMemory, MultimodalTemporalTracker,
    run_autonomous_pipeline
)

def render_camera_only_frame(res, output_path):
    """Renders only the Camera Vision view with LiDAR projection and tracked obstacles."""
    cam_image = res['cam_image']
    u_proj = res['u_proj']
    v_proj = res['v_proj']
    depths_proj = res['depths_proj']
    obstacles = res['tracked_obstacles']
    sample_idx = res['sample_idx']
    ego_speed_mps = res['ego_speed_mps']
    rec_speed_kmh = res['rec_speed_kmh']

    fig = plt.figure(figsize=(16, 10), facecolor='#0B0F19', dpi=120)
    
    # Top scenario header
    suptitle_text = (
        f"Scenario Keyframe: #{sample_idx} | Ego Speed: {ego_speed_mps*3.6:.1f} km/h | Target Speed: {rec_speed_kmh:.1f} km/h"
    )
    fig.suptitle(suptitle_text, fontsize=15, fontweight='bold', color='#58A6FF', y=0.96)

    ax = fig.add_axes([0.04, 0.04, 0.92, 0.86])
    ax.imshow(cam_image)

    # Overlaid LiDAR depth points
    if len(u_proj) > 0:
        scatter = ax.scatter(u_proj, v_proj, c=depths_proj, cmap='turbo', s=2.5, alpha=0.7)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.015)
        cbar.set_label('LiDAR Depth (meters)', color='white', fontsize=11, fontweight='bold')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Overlaid Tracked Bounding Boxes
    for obs in obstacles:
        ub, vb = obs.get("u_box"), obs.get("v_box")
        tid = obs.get("track_id", "")
        if ub is not None and vb is not None and 50 < ub < cam_image.width - 50 and 50 < vb < cam_image.height - 50:
            bh = int(1200 / max(obs["dist"], 5.0))
            bw = int(bh * 0.75)
            rect = patches.Rectangle((ub - bw/2, vb - bh/2), bw, bh,
                                     linewidth=2.0, edgecolor='#00FF66', facecolor='none')
            ax.add_patch(rect)
            ax.text(ub - bw/2, vb - bh/2 - 8,
                     f"ID#{tid} {obs['class']} ({obs['dist']:.1f}m)",
                     color='black', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="square,pad=0.2", facecolor='#00FF66', edgecolor='none', alpha=0.85))

    ax.set_title("1. Camera Vision: RGB Frame + Calibrated LiDAR Depth Overlay & Tracked Detections",
                 color='white', fontsize=13, fontweight='bold', pad=10)
    ax.axis('off')

    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()


def generate_camera_video(scene_idx=0, num_frames=5, output_video=r"D:\PPD2\camera_vision_simulation.mp4"):
    print("=" * 75)
    print(f"   GENERATING CAMERA VISION ONLY VIDEO (Scene #{scene_idx}, {num_frames} Frames)")
    print("=" * 75)

    with open(os.path.join(META_DIR, "scene.json")) as f:
        scenes = json.load(f)
    with open(os.path.join(META_DIR, "sample.json")) as f:
        samples = json.load(f)

    if scene_idx >= len(scenes):
        scene_idx = 0
    sc = scenes[scene_idx]
    sample_token_to_idx = {s["token"]: idx for idx, s in enumerate(samples)}

    # Sequence chain
    curr_token = sc["first_sample_token"]
    frame_indices = []
    while curr_token and len(frame_indices) < num_frames:
        idx = sample_token_to_idx.get(curr_token)
        if idx is not None:
            frame_indices.append(idx)
            curr_token = samples[idx]["next"]
        else:
            break

    print(f"[SCENE] {sc['name']} | Context: {sc['description']}")
    print(f"[FRAMES] Processing sequence chain: {frame_indices}")

    ego_memory = TemporalEgoMemory()
    tracker = MultimodalTemporalTracker(max_age=2, dist_threshold=3.5)

    temp_image_files = []
    for step, s_idx in enumerate(frame_indices):
        print(f"\n>> Processing Frame {step+1}/{len(frame_indices)} (Keyframe #{s_idx})...")
        res = run_autonomous_pipeline(
            sample_idx=s_idx,
            save_dashboard=False, # We render custom camera-only view
            ego_memory=ego_memory,
            tracker=tracker
        )
        frame_img_path = os.path.join(BASE_DIR, f"cam_only_step_{step+1}.png")
        render_camera_only_frame(res, frame_img_path)
        temp_image_files.append(frame_img_path)
        print(f"   Rendered: {frame_img_path}")

    # Build concat file for FFmpeg
    concat_txt = os.path.join(BASE_DIR, "cam_frames_list.txt")
    with open(concat_txt, "w") as f:
        for img in temp_image_files:
            clean_p = img.replace('\\', '/')
            f.write(f"file '{clean_p}'\n")
            f.write("duration 1.5\n")
        # Final image repeat for concat demuxer
        last_p = temp_image_files[-1].replace('\\', '/')
        f.write(f"file '{last_p}'\n")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if os.path.exists(output_video):
        os.remove(output_video)

    # Encode with H.264, yuv420p, even dimensions
    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-profile:v", "main",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-r", "24",
        output_video
    ]

    print("\n[ENCODING] Re-encoding video with FFmpeg libx264...")
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(output_video) / 1e6
    print(f"\n[SUCCESS] Camera Vision Video Created: {output_video} ({size_mb:.2f} MB)")

    # Also generate an animated GIF of camera only
    gif_path = os.path.join(BASE_DIR, "camera_vision_simulation.gif")
    gif_frames = [Image.open(f) for f in temp_image_files]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1500,
        loop=0
    )
    print(f"[SUCCESS] Camera Vision GIF Created: {gif_path} ({os.path.getsize(gif_path)/1e6:.2f} MB)")

    return output_video, gif_path

if __name__ == "__main__":
    generate_camera_video(scene_idx=0, num_frames=5)

