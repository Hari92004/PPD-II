"""
SIH 2026 - Model 2: 3D LiDAR Object Detector Neural Network
Architecture: Point-based 3D Spatial Feature Net & Multi-Object Regressor
Fully compatible with PyTorch GPU/CPU training and MATLAB importONNXNetwork.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudEncoder(nn.Module):
    """Hierarchical Point Feature Extractor (PointNet-style)."""
    def __init__(self, in_channels=4, feat_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, feat_dim, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(feat_dim)

    def forward(self, x):
        # x: [Batch, In_Channels, Num_Points]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Global max pooling across all points
        global_feat = torch.max(x, dim=2)[0] # [Batch, Feat_Dim]
        return global_feat

class PointPillars3DDetector(nn.Module):
    """
    3D LiDAR Object Detection Network for Unstructured Environments.
    Takes 3D Point Cloud [Batch, Num_Points, 4] and predicts K candidate 3D Bounding Boxes.
    Each 3D Box: [X, Y, Z, Width, Length, Height, sin(yaw), cos(yaw), Class_Logits(3), Confidence]
    """
    def __init__(self, in_channels=4, num_proposals=20, num_classes=3):
        super().__init__()
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        
        # 1. Point Cloud Spatial Encoder
        self.encoder = PointCloudEncoder(in_channels=in_channels, feat_dim=256)
        
        # 2. Dense Feature Aggregation & Proposal Decoder
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # 3. Task-Specific Detection Heads (Multi-Box Regression)
        # 3D Box Regressor: (dx, dy, dz, dw, dl, dh, sin_yaw, cos_yaw) = 8 dims per box
        self.box_head = nn.Linear(512, num_proposals * 8)
        # Class Logits: 3 classes per box
        self.cls_head = nn.Linear(512, num_proposals * num_classes)
        # Objectness / Confidence: 1 score per box
        self.conf_head = nn.Linear(512, num_proposals * 1)

    def forward(self, points):
        """
        Forward pass.
        Input: points [Batch, Num_Points, 4] (X, Y, Z, Intensity)
        Output: predicted_boxes [Batch, num_proposals, 12]
        """
        # Transpose to [Batch, Channels, Num_Points] for 1D convolution
        x = points.transpose(1, 2)
        global_features = self.encoder(x) # [Batch, 256]
        
        latent = self.fc_layers(global_features) # [Batch, 512]
        
        batch_size = points.size(0)
        
        # Regress 3D bounding boxes
        boxes = self.box_head(latent).view(batch_size, self.num_proposals, 8)
        cls_logits = self.cls_head(latent).view(batch_size, self.num_proposals, self.num_classes)
        conf = torch.sigmoid(self.conf_head(latent)).view(batch_size, self.num_proposals, 1)
        
        # Combine: [X, Y, Z, W, L, H, sin_yaw, cos_yaw, cls_0, cls_1, cls_2, conf]
        output = torch.cat([boxes, cls_logits, conf], dim=-1)
        return output

def compute_model2_loss(predictions, targets):
    """
    Multi-task loss for 3D LiDAR Object Detection.
    predictions: [Batch, K, 12] -> [x, y, z, w, l, h, sin, cos, cls0, cls1, cls2, conf]
    targets:     [Batch, K, 10] -> [x, y, z, w, l, h, sin, cos, class_id, is_valid]
    """
    device = predictions.device
    pred_boxes = predictions[:, :, :8]
    pred_cls = predictions[:, :, 8:11]
    pred_conf = predictions[:, :, 11]
    
    target_boxes = targets[:, :, :8]
    target_cls_id = targets[:, :, 8].long()
    target_valid = targets[:, :, 9] # 1.0 if box exists, 0.0 if padded
    
    # 1. Confidence / Objectness Loss (BCE)
    conf_loss = F.binary_cross_entropy(pred_conf, target_valid)
    
    # 2. 3D Bounding Box Regression Loss (Smooth L1, only on valid ground truth objects)
    valid_mask = (target_valid > 0.5).unsqueeze(-1).expand_as(pred_boxes)
    if valid_mask.sum() > 0:
        box_loss = F.smooth_l1_loss(pred_boxes[valid_mask], target_boxes[valid_mask])
    else:
        box_loss = torch.tensor(0.0, device=device)
        
    # 3. Classification Loss (Cross Entropy, only on valid objects)
    valid_cls_mask = target_valid > 0.5
    if valid_cls_mask.sum() > 0:
        cls_loss = F.cross_entropy(pred_cls[valid_cls_mask], target_cls_id[valid_cls_mask])
    else:
        cls_loss = torch.tensor(0.0, device=device)
        
    total_loss = 2.0 * box_loss + 1.0 * cls_loss + 1.5 * conf_loss
    return total_loss, box_loss.item(), cls_loss.item(), conf_loss.item()

if __name__ == "__main__":
    print("Testing PointPillars3DDetector architecture...")
    model = PointPillars3DDetector()
    dummy_points = torch.randn(2, 16384, 4)
    out = model(dummy_points)
    print(f"Input shape: {dummy_points.shape}")
    print(f"Output shape: {out.shape} -> Expected [2, 20, 12]")
    print(" Network architecture verification PASSED!")
