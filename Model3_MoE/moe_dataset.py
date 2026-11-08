"""
====================================================================================================
Model 3 MoE Trajectory Prediction Dataset Loader (with Truck/Bus Class Balancing)
====================================================================================================
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from scipy.interpolate import interp1d

CLASSES = [
    'auto_rickshaw',
    'cow_cattle',
    'pedestrian',
    'two_wheeler',
    'pothole',
    'truck_bus',
    'car'
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

def map_nuscenes_category(raw_cat):
    raw = raw_cat.lower()
    if 'pedestrian' in raw or 'human' in raw:
        return 'pedestrian'
    elif 'motorcycle' in raw or 'bicycle' in raw:
        return 'two_wheeler'
    elif any(k in raw for k in ['truck', 'bus', 'trailer', 'construction']):
        return 'truck_bus'
    elif 'car' in raw:
        return 'car'
    elif 'rickshaw' in raw:
        return 'auto_rickshaw'
    elif 'animal' in raw:
        return 'cow_cattle'
    return None

class MoETrajectoryDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        past = sample['past'].copy()     # [8, 2]
        future = sample['future'].copy() # [12, 2]
        class_id = sample['class_id']

        if self.augment:
            # 1. Random 2D Rotation [-pi, pi]
            theta = np.random.uniform(-np.pi, np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            past = past @ rot.T
            future = future @ rot.T

            # 2. Random Horizontal Mirror Flip (p=0.5)
            if np.random.rand() > 0.5:
                past[:, 1] = -past[:, 1]
                future[:, 1] = -future[:, 1]

            # 3. Small Gaussian Noise (sigma=0.03m)
            noise = np.random.normal(0.0, 0.03, size=past.shape).astype(np.float32)
            past += noise

            # 4. Truck/Bus specific kinematic scaling (speed variation 0.9x to 1.1x)
            if class_id == CLASS_TO_IDX['truck_bus']:
                scale = np.random.uniform(0.9, 1.1)
                past = past * scale
                future = future * scale

        return {
            'past_pos': torch.tensor(past, dtype=torch.float32),
            'future_pos': torch.tensor(future, dtype=torch.float32),
            'class_id': torch.tensor(class_id, dtype=torch.long)
        }


def extract_nuscenes_tracks(nuscenes_meta_dir):
    with open(os.path.join(nuscenes_meta_dir, "scene.json")) as f:
        scenes = json.load(f)
    with open(os.path.join(nuscenes_meta_dir, "sample.json")) as f:
        samples = json.load(f)
    with open(os.path.join(nuscenes_meta_dir, "sample_annotation.json")) as f:
        annotations = json.load(f)
    with open(os.path.join(nuscenes_meta_dir, "instance.json")) as f:
        instances = json.load(f)
    with open(os.path.join(nuscenes_meta_dir, "category.json")) as f:
        categories = json.load(f)

    cat_map = {c["token"]: c["name"] for c in categories}
    inst_cat = {inst["token"]: cat_map[inst["category_token"]] for inst in instances}
    sample_map = {s["token"]: s for s in samples}

    inst_tracks = defaultdict(list)
    for ann in annotations:
        inst_token = ann["instance_token"]
        s = sample_map[ann["sample_token"]]
        t_sec = s["timestamp"] / 1e6
        x, y, _ = ann["translation"]
        cname = inst_cat[inst_token]
        inst_tracks[inst_token].append({
            "timestamp": t_sec,
            "x": x,
            "y": y,
            "category": cname,
            "scene_token": s["scene_token"]
        })
    return scenes, inst_tracks


def generate_synthetic_indian_trajectories(num_samples=3000, dt=0.2, seed=42):
    rng = np.random.RandomState(seed)
    synthetic_windows = []
    classes = ['auto_rickshaw', 'cow_cattle', 'pothole']
    samples_per_class = num_samples // len(classes)

    for cname in classes:
        cid = CLASS_TO_IDX[cname]
        for _ in range(samples_per_class):
            total_steps = 20
            traj = []
            cx, cy = 0.0, 0.0

            if cname == 'auto_rickshaw':
                speed = rng.uniform(3.5, 8.5)
                heading = rng.uniform(-np.pi, np.pi)
                freq = rng.uniform(0.6, 1.4)
                for step in range(total_steps):
                    lat = 0.6 * np.sin(freq * step * dt)
                    cx += speed * np.cos(heading) * dt + lat * (-np.sin(heading)) * dt
                    cy += speed * np.sin(heading) * dt + lat * (np.cos(heading)) * dt
                    traj.append([cx, cy])

            elif cname == 'cow_cattle':
                speed = rng.uniform(0.5, 1.6)
                heading = rng.uniform(-np.pi, np.pi)
                for step in range(total_steps):
                    if step >= 7 and rng.rand() > 0.4:
                        heading += rng.uniform(-0.6, 0.6)
                    cx += speed * np.cos(heading) * dt + rng.normal(0, 0.02)
                    cy += speed * np.sin(heading) * dt + rng.normal(0, 0.02)
                    traj.append([cx, cy])

            elif cname == 'pothole':
                ego_speed = rng.uniform(5.0, 12.0)
                heading = np.pi
                for _ in range(total_steps):
                    cx += ego_speed * np.cos(heading) * dt
                    cy += ego_speed * np.sin(heading) * dt + rng.normal(0, 0.01)
                    traj.append([cx, cy])

            traj = np.array(traj, dtype=np.float32)
            past = traj[:8]
            future = traj[8:20]
            synthetic_windows.append({
                'past': past,
                'future': future,
                'class_id': cid,
                'category': cname,
                'scene_token': 'synthetic_scene'
            })
    return synthetic_windows


def build_moe_datasets(nuscenes_meta_dir=r"D:\PPD2\dataset mini\v1.0-mini\v1.0-mini",
                       val_scene_ratio=0.2, seed=42, oversample_truck=True):
    np.random.seed(seed)
    scenes, inst_tracks = extract_nuscenes_tracks(nuscenes_meta_dir)

    scene_tokens = [s["token"] for s in scenes]
    np.random.shuffle(scene_tokens)
    num_val_scenes = max(1, int(len(scene_tokens) * val_scene_ratio))
    val_scenes = set(scene_tokens[:num_val_scenes])
    train_scenes = set(scene_tokens[num_val_scenes:])

    train_windows = []
    val_windows = []
    class_counts_train = defaultdict(int)
    class_counts_val = defaultdict(int)

    for inst_token, track in inst_tracks.items():
        if len(track) < 5:
            continue
        track.sort(key=lambda p: p["timestamp"])
        cname = map_nuscenes_category(track[0]["category"])
        if cname is None:
            continue
        cid = CLASS_TO_IDX[cname]

        ts = np.array([p["timestamp"] for p in track])
        xs = np.array([p["x"] for p in track])
        ys = np.array([p["y"] for p in track])

        if ts[-1] - ts[0] < 3.8:
            continue

        target_ts = np.arange(ts[0], ts[-1], 0.2)
        if len(target_ts) < 20:
            continue

        fx = interp1d(ts, xs, kind='linear')
        fy = interp1d(ts, ys, kind='linear')
        interp_xs = fx(target_ts)
        interp_ys = fy(target_ts)
        traj_5hz = np.column_stack([interp_xs, interp_ys])

        scene_token = track[0]["scene_token"]
        is_val = scene_token in val_scenes

        for i in range(len(traj_5hz) - 20 + 1):
            w = traj_5hz[i:i+20]
            item = {
                'past': w[:8].astype(np.float32),
                'future': w[8:20].astype(np.float32),
                'class_id': cid,
                'category': cname,
                'scene_token': scene_token
            }
            if is_val:
                val_windows.append(item)
                class_counts_val[cname] += 1
            else:
                train_windows.append(item)
                class_counts_train[cname] += 1
                # Oversample truck_bus in training set so network learns heavy vehicle dynamics thoroughly
                if oversample_truck and cname == 'truck_bus':
                    train_windows.append(item)
                    class_counts_train[cname] += 1

    # Synthetic Indian road data
    synth_samples = generate_synthetic_indian_trajectories(num_samples=3000, seed=seed)
    np.random.shuffle(synth_samples)
    n_val_synth = int(len(synth_samples) * val_scene_ratio)
    val_synth = synth_samples[:n_val_synth]
    train_synth = synth_samples[n_val_synth:]

    for item in train_synth:
        train_windows.append(item)
        class_counts_train[item['category']] += 1
    for item in val_synth:
        val_windows.append(item)
        class_counts_val[item['category']] += 1

    total_windows = len(train_windows) + len(val_windows)
    print("\n" + "=" * 70)
    print("   MODEL 3 MoE DATASET (TRUCK/BUS BALANCED)")
    print("=" * 70)
    print(f"Total Windows: {total_windows:,} | Train: {len(train_windows):,} | Val: {len(val_windows):,}")
    print("\nPer-Class Distribution:")
    for cname in CLASSES:
        n_tr = class_counts_train[cname]
        n_val = class_counts_val[cname]
        print(f"  [{cname:<14}] Train: {n_tr:6,d} | Val: {n_val:5,d} | Total: {n_tr + n_val:6,d}")
    print("=" * 70)

    train_dataset = MoETrajectoryDataset(train_windows, augment=True)
    val_dataset = MoETrajectoryDataset(val_windows, augment=False)
    return train_dataset, val_dataset
