"""
SIH 2026 - Model 2 (3D LiDAR Perception)
Dataset Loader for nuScenes v1.0-mini
Reads binary .pcd.bin point clouds and extracts calibrated 3D bounding boxes.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Define Class Mappings
CLASS_NAMES = ['vehicle', 'pedestrian', 'cyclist']
CLASS_MAP = {
    'vehicle.car': 0,
    'vehicle.truck': 0,
    'vehicle.bus.bendy': 0,
    'vehicle.bus.rigid': 0,
    'vehicle.construction': 0,
    'vehicle.emergency.ambulance': 0,
    'vehicle.emergency.police': 0,
    'vehicle.trailer': 0,
    'human.pedestrian.adult': 1,
    'human.pedestrian.child': 1,
    'human.pedestrian.construction_worker': 1,
    'human.pedestrian.police_officer': 1,
    'vehicle.bicycle': 2,
    'vehicle.motorcycle': 2
}

def quaternion_to_rotation_matrix(q):
    """Converts a quaternion [w, x, y, z] to a 3x3 orthonormal rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)

def read_point_cloud_bin(file_path):
    """
    Reads nuScenes 32-beam LiDAR binary point cloud (.pcd.bin).
    Each point has 5 float32 channels: [x, y, z, intensity, ring_index].
    Returns Nx4 array [x, y, z, intensity].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")
    raw_points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)
    return raw_points[:, :4] # x, y, z, intensity

class NuScenesLidarDataset(Dataset):
    """
    PyTorch Dataset for nuScenes LiDAR 3D Object Detection.
    Preprocesses point clouds into ROI and transforms global 3D BBoxes to LiDAR frame.
    """
    def __init__(self, data_root=None,
                 roi_x=(-40.0, 40.0), roi_y=(-40.0, 40.0), roi_z=(-3.0, 3.0),
                 max_points=16384, max_boxes=20):
        if data_root is None:
            if os.path.exists(r"D:\PPD2\dataset mini\v1.0-mini"):
                data_root = r"D:\PPD2\dataset mini\v1.0-mini"
            else:
                data_root = r"D:\SIH26\dataset mini\v1.0-mini"
        self.data_root = data_root
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_z = roi_z
        self.max_points = max_points
        self.max_boxes = max_boxes

        json_dir = os.path.join(data_root, "v1.0-mini")
        print(f"[DATASET] Loading nuScenes metadata from: {json_dir}")

        with open(os.path.join(json_dir, "sample.json"), 'r') as f:
            self.samples = json.load(f)
        with open(os.path.join(json_dir, "sample_data.json"), 'r') as f:
            sample_data_list = json.load(f)
            self.sample_data = {sd['token']: sd for sd in sample_data_list}
        with open(os.path.join(json_dir, "sample_annotation.json"), 'r') as f:
            annotations_list = json.load(f)
        with open(os.path.join(json_dir, "calibrated_sensor.json"), 'r') as f:
            self.calibrated_sensors = {cs['token']: cs for cs in json.load(f)}
        with open(os.path.join(json_dir, "ego_pose.json"), 'r') as f:
            self.ego_poses = {ep['token']: ep for ep in json.load(f)}
        with open(os.path.join(json_dir, "category.json"), 'r') as f:
            self.categories = {c['token']: c['name'] for c in json.load(f)}
        with open(os.path.join(json_dir, "instance.json"), 'r') as f:
            self.instances = {inst['token']: inst['category_token'] for inst in json.load(f)}

        # Group annotations by sample token
        self.annotations_by_sample = {}
        for ann in annotations_list:
            s_token = ann['sample_token']
            if s_token not in self.annotations_by_sample:
                self.annotations_by_sample[s_token] = []
            self.annotations_by_sample[s_token].append(ann)

        # Index LIDAR_TOP samples from sample_data.json
        self.lidar_samples = []
        for sd in sample_data_list:
            if 'LIDAR_TOP' in sd['filename'] and sd['is_key_frame']:
                self.lidar_samples.append({
                    'sample_token': sd['sample_token'],
                    'lidar_sd': sd
                })

        print(f"[SUCCESS] Loaded {len(self.lidar_samples)} keyframe LiDAR point clouds from nuScenes.")

    def __len__(self):
        return len(self.lidar_samples)

    def _transform_box_to_lidar(self, ann, ego_pose, calib_sensor):
        """Transforms 3D bounding box from global frame to LiDAR frame."""
        box_pos_global = np.array(ann['translation'], dtype=np.float32)
        box_rot_global = quaternion_to_rotation_matrix(ann['rotation'])
        box_size = np.array(ann['size'], dtype=np.float32) # [width, length, height]

        # 1. Global to Ego
        ego_trans = np.array(ego_pose['translation'], dtype=np.float32)
        r_ego = quaternion_to_rotation_matrix(ego_pose['rotation'])
        pos_ego = r_ego.T @ (box_pos_global - ego_trans)
        rot_ego = r_ego.T @ box_rot_global

        # 2. Ego to LiDAR
        lidar_trans = np.array(calib_sensor['translation'], dtype=np.float32)
        r_lidar = quaternion_to_rotation_matrix(calib_sensor['rotation'])
        pos_lidar = r_lidar.T @ (pos_ego - lidar_trans)
        rot_lidar = r_lidar.T @ rot_ego

        # Extract yaw (heading angle in LiDAR XY plane)
        yaw = np.arctan2(rot_lidar[1, 0], rot_lidar[0, 0])

        return pos_lidar, box_size, yaw

    def __getitem__(self, idx):
        item = self.lidar_samples[idx]
        sd = item['lidar_sd']
        sample_token = item['sample_token']

        # Load raw point cloud
        pcd_rel_path = sd['filename'].replace('/', os.sep)
        pcd_path = os.path.join(self.data_root, pcd_rel_path)
        points = read_point_cloud_bin(pcd_path) # [N, 4]

        # Filter ROI
        mask = (points[:, 0] >= self.roi_x[0]) & (points[:, 0] <= self.roi_x[1]) & \
               (points[:, 1] >= self.roi_y[0]) & (points[:, 1] <= self.roi_y[1]) & \
               (points[:, 2] >= self.roi_z[0]) & (points[:, 2] <= self.roi_z[1])
        points = points[mask]

        # Standardize point count (sample / pad)
        n_pts = points.shape[0]
        if n_pts >= self.max_points:
            choice = np.random.choice(n_pts, self.max_points, replace=False)
            points_tensor = points[choice]
        else:
            pad = np.zeros((self.max_points - n_pts, 4), dtype=np.float32)
            points_tensor = np.vstack([points, pad])

        # Calibrated 3D Bounding Boxes
        calib_sensor = self.calibrated_sensors[sd['calibrated_sensor_token']]
        ego_pose = self.ego_poses[sd['ego_pose_token']]
        annotations = self.annotations_by_sample.get(sample_token, [])

        boxes_list = []
        for ann in annotations:
            cat_token = self.instances.get(ann.get('instance_token', ''), '')
            cat_name = self.categories.get(cat_token, '')
            class_id = CLASS_MAP.get(cat_name, -1)
            if class_id == -1:
                continue # Skip irrelevant classes (e.g. barriers, debris)

            pos, size, yaw = self._transform_box_to_lidar(ann, ego_pose, calib_sensor)

            # Check if box center is inside ROI
            if (self.roi_x[0] <= pos[0] <= self.roi_x[1] and
                self.roi_y[0] <= pos[1] <= self.roi_y[1] and
                self.roi_z[0] <= pos[2] <= self.roi_z[1]):
                # Target format: [x, y, z, width, length, height, sin(yaw), cos(yaw), class_id, confidence=1.0]
                boxes_list.append([
                    pos[0], pos[1], pos[2],
                    size[0], size[1], size[2],
                    np.sin(yaw), np.cos(yaw),
                    float(class_id), 1.0
                ])

        # Pad or trim boxes to fixed max_boxes
        target_boxes = np.zeros((self.max_boxes, 10), dtype=np.float32)
        n_boxes = min(len(boxes_list), self.max_boxes)
        if n_boxes > 0:
            target_boxes[:n_boxes] = np.array(boxes_list[:n_boxes], dtype=np.float32)

        return (torch.from_numpy(points_tensor.astype(np.float32)),
                torch.from_numpy(target_boxes))

def get_lidar_dataloader(batch_size=8, shuffle=True):
    dataset = NuScenesLidarDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataset, loader

if __name__ == "__main__":
    print("Testing NuScenesLidarDataset loader...")
    ds = NuScenesLidarDataset()
    pts, boxes = ds[0]
    print(f"Sample 0 points shape: {pts.shape}, dtype: {pts.dtype}")
    print(f"Sample 0 target boxes shape: {boxes.shape}")
    num_valid_boxes = int((boxes[:, 9] > 0).sum())
    print(f"Valid 3D Ground Truth Bounding Boxes: {num_valid_boxes}")
    if num_valid_boxes > 0:
        first_box = boxes[0].numpy()
        print(f"First Box: X={first_box[0]:.2f}m, Y={first_box[1]:.2f}m, Z={first_box[2]:.2f}m, "
              f"W={first_box[3]:.2f}m, L={first_box[4]:.2f}m, H={first_box[5]:.2f}m, "
              f"Class={CLASS_NAMES[int(first_box[8])]}")
    print("[SUCCESS] Dataset loader test PASSED!")
