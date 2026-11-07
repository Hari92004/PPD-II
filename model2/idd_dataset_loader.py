"""
SIH 2026 - Model 2 (3D LiDAR Perception)
Dataset Loader for IDD Multimodal with 3D Pseudo-Labels & Spatial Augmentations.
"""

import os
import io
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IDDLidarDataset(Dataset):
    """
    Dataset streaming point clouds directly from idd_mm_supplement.zip
    paired with precomputed pseudo-labels and online 3D data augmentations.
    """
    def __init__(
        self,
        zip_path=r"D:\user\Downloads\idd_mm_supplement.zip",
        labels_npz=r"D:\PPD2\model2\idd_pseudo_labels_clean.npz",
        roi_x=(-40.0, 40.0),
        roi_y=(-40.0, 40.0),
        roi_z=(-3.0, 3.0),
        max_points=16384,
        max_boxes=20,
        augment=False
    ):
        self.zip_path = zip_path
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_z = roi_z
        self.max_points = max_points
        self.max_boxes = max_boxes
        self.augment = augment
        
        if not os.path.exists(labels_npz):
            raise FileNotFoundError(f"Pseudo-labels file not found at: {labels_npz}")
            
        data = np.load(labels_npz, allow_pickle=True)
        self.files = data['files']
        self.targets = data['targets'] # [N, 20, 10]
        self._zf = None

    def _get_zip(self):
        if self._zf is None and os.path.exists(self.zip_path):
            self._zf = zipfile.ZipFile(self.zip_path, 'r')
        return self._zf

    def __len__(self):
        return len(self.files)

    def _apply_augmentation(self, points, target_boxes):
        """Applies 3D Yaw Rotation, Scaling, and Jitter to points and 3D boxes."""
        # 1. Random Yaw Rotation [-15 deg, +15 deg]
        angle = np.random.uniform(-np.pi / 12, np.pi / 12)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_mat = np.array([
            [cos_a, -sin_a, 0.0],
            [sin_a,  cos_a, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float32)

        points[:, :3] = points[:, :3] @ rot_mat.T

        # 2. Random Scale [0.95, 1.05]
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale

        # Transform valid target boxes
        aug_boxes = target_boxes.copy()
        for i in range(aug_boxes.shape[0]):
            if aug_boxes[i, 9] > 0.5: # valid box
                # Rotate center
                pos = aug_boxes[i, :3] @ rot_mat.T
                aug_boxes[i, :3] = pos * scale
                # Scale dimensions
                aug_boxes[i, 3:6] *= scale
                # Rotate yaw angle
                old_sin, old_cos = aug_boxes[i, 6], aug_boxes[i, 7]
                old_yaw = np.arctan2(old_sin, old_cos)
                new_yaw = old_yaw + angle
                aug_boxes[i, 6] = np.sin(new_yaw)
                aug_boxes[i, 7] = np.cos(new_yaw)

        return points, aug_boxes

    def __getitem__(self, idx):
        # Support reading either from extracted dataset folder or fallback to zip
        f_name = str(self.files[idx])
        parts = f_name.split('/')
        if len(parts) >= 4:
            drive = parts[3]
            frame = parts[4]
            direct_path = os.path.join(r"D:\PPD2\dataset_idd_synchronized\lidar", f"{drive}_lidar_{frame}")
        else:
            direct_path = ""

        if direct_path and os.path.exists(direct_path):
            arr = np.load(direct_path).astype(np.float32)
        else:
            zf = self._get_zip()
            raw_bytes = zf.read(f_name)
            arr = np.load(io.BytesIO(raw_bytes)).astype(np.float32)

        # 1. Filter ROI
        mask = (arr[:, 0] >= self.roi_x[0]) & (arr[:, 0] <= self.roi_x[1]) & \
               (arr[:, 1] >= self.roi_y[0]) & (arr[:, 1] <= self.roi_y[1]) & \
               (arr[:, 2] >= self.roi_z[0]) & (arr[:, 2] <= self.roi_z[1])
        points = arr[mask][:, :4] # [x, y, z, intensity]
        target_boxes = self.targets[idx].copy()

        # 2. Online Augmentation (training split only)
        if self.augment and np.random.rand() > 0.5:
            points, target_boxes = self._apply_augmentation(points, target_boxes)

        # 3. Standardize point count
        n_pts = points.shape[0]
        if n_pts >= self.max_points:
            choice = np.random.choice(n_pts, self.max_points, replace=False)
            points_tensor = points[choice]
        elif n_pts > 0:
            choice = np.random.choice(n_pts, self.max_points, replace=True)
            points_tensor = points[choice]
        else:
            points_tensor = np.zeros((self.max_points, 4), dtype=np.float32)

        return (
            torch.from_numpy(points_tensor.astype(np.float32)),
            torch.from_numpy(target_boxes.astype(np.float32))
        )

def get_idd_dataloaders(
    zip_path=r"D:\user\Downloads\idd_mm_supplement.zip",
    labels_npz=r"D:\PPD2\model2\idd_pseudo_labels_clean.npz",
    batch_size=8,
    val_ratio=0.2,
    augment=True
):
    train_dataset = IDDLidarDataset(zip_path=zip_path, labels_npz=labels_npz, augment=augment)
    val_dataset = IDDLidarDataset(zip_path=zip_path, labels_npz=labels_npz, augment=False)
    
    total_samples = len(train_dataset)
    val_size = int(val_ratio * total_samples)
    train_size = total_samples - val_size

    # Deterministic split indices
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_sub = torch.utils.data.Subset(train_dataset, train_indices)
    val_sub = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_size, val_size
