"""
SIH 2026 - High-Precision Pseudo-Label Generator (Clean Threshold tau >= 0.65)
Filters out weak/ghost candidate proposals from the Teacher model
to provide high-fidelity 3D obstacle targets for IDD Multimodal point clouds.
"""

import os
import io
import time
import zipfile
import numpy as np
import torch

from pointpillars_net import PointPillars3DDetector

def generate_clean_pseudo_labels(
    zip_path=r"D:\user\Downloads\idd_mm_supplement.zip",
    teacher_ckpt=r"D:\PPD2\model2\best_model2_lidar.pth",
    output_npz=r"D:\PPD2\model2\idd_pseudo_labels_clean.npz",
    max_samples=2500,
    conf_thresh=0.65, # High precision threshold
    num_points=16384,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print("=" * 65)
    print("   GENERATING HIGH-PRECISION IDD 3D PSEUDO-LABELS (TAU >= 0.65)")
    print("=" * 65)
    
    # 1. Load Teacher Model
    model = PointPillars3DDetector(in_channels=4, num_proposals=20, num_classes=3).to(device)
    ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    # 2. Select Frames
    zf = zipfile.ZipFile(zip_path, 'r')
    lidar_files = [f for f in zf.namelist() if f.endswith('.npy')]
    lidar_files.sort()
    
    if max_samples and max_samples < len(lidar_files):
        indices = np.linspace(0, len(lidar_files) - 1, max_samples, dtype=int)
        selected_files = [lidar_files[i] for i in indices]
    else:
        selected_files = lidar_files

    print(f"[PREVIEW] Processing {len(selected_files)} point cloud frames on {device}...")
    roi_x, roi_y, roi_z = (-40.0, 40.0), (-40.0, 40.0), (-3.0, 3.0)
    
    valid_file_list = []
    targets_list = []
    
    start_time = time.time()
    n_batches = (len(selected_files) + batch_size - 1) // batch_size
    
    for b_idx in range(n_batches):
        batch_files = selected_files[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch_tensors = []
        batch_valid_names = []

        for f_name in batch_files:
            try:
                raw_bytes = zf.read(f_name)
                arr = np.load(io.BytesIO(raw_bytes)).astype(np.float32)
                
                mask = (arr[:, 0] >= roi_x[0]) & (arr[:, 0] <= roi_x[1]) & \
                       (arr[:, 1] >= roi_y[0]) & (arr[:, 1] <= roi_y[1]) & \
                       (arr[:, 2] >= roi_z[0]) & (arr[:, 2] <= roi_z[1])
                pts = arr[mask][:, :4]
                
                n_pts = pts.shape[0]
                if n_pts >= num_points:
                    choice = np.random.choice(n_pts, num_points, replace=False)
                    pts_std = pts[choice]
                elif n_pts > 0:
                    choice = np.random.choice(n_pts, num_points, replace=True)
                    pts_std = pts[choice]
                else:
                    pts_std = np.zeros((num_points, 4), dtype=np.float32)
                    
                batch_tensors.append(pts_std)
                batch_valid_names.append(f_name)
            except Exception as e:
                pass
                
        if not batch_tensors:
            continue
            
        tensor_gpu = torch.from_numpy(np.array(batch_tensors)).to(device)
        with torch.no_grad():
            preds = model(tensor_gpu) # [B, 20, 12]
            
        preds_cpu = preds.cpu().numpy()
        
        for i, f_name in enumerate(batch_valid_names):
            pred_i = preds_cpu[i] # [20, 12]
            boxes = pred_i[:, :8]
            cls_logits = pred_i[:, 8:11]
            classes = np.argmax(cls_logits, axis=-1)
            confs = pred_i[:, 11]
            
            # Apply 3D-NMS directly to teacher pseudo-labels to prevent duplicate boxes
            valid_mask = confs >= conf_thresh
            if np.sum(valid_mask) > 1:
                cand_idx = np.where(valid_mask)[0]
                cand_idx = cand_idx[np.argsort(-confs[cand_idx])]
                keep = []
                for c in cand_idx:
                    c_pos = boxes[c, :2]
                    suppress = False
                    for k in keep:
                        if np.linalg.norm(c_pos - boxes[k, :2]) < 1.8:
                            suppress = True
                            break
                    if not suppress:
                        keep.append(c)
                valid_mask = np.zeros(20, dtype=bool)
                valid_mask[keep] = True

            target = np.zeros((20, 10), dtype=np.float32)
            n_valid = np.sum(valid_mask)
            if n_valid > 0:
                v_boxes = boxes[valid_mask]
                v_classes = classes[valid_mask]
                target[:n_valid, :8] = v_boxes
                target[:n_valid, 8] = v_classes.astype(np.float32)
                target[:n_valid, 9] = 1.0 # is_valid
                
            valid_file_list.append(f_name)
            targets_list.append(target)
            
        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == n_batches:
            fps = len(valid_file_list) / max(time.time() - start_time, 0.001)
            print(f"[{b_idx + 1}/{n_batches}] Processed {len(valid_file_list)} frames ({fps:.1f} fps)...")

    zf.close()
    
    file_array = np.array(valid_file_list)
    targets_array = np.array(targets_list, dtype=np.float32)
    np.savez_compressed(output_npz, files=file_array, targets=targets_array)
    file_mb = os.path.getsize(output_npz) / 1e6
    print(f"[SUCCESS] Clean pseudo-labels saved to: {output_npz} ({file_mb:.2f} MB)")
    return output_npz

if __name__ == "__main__":
    generate_clean_pseudo_labels()
