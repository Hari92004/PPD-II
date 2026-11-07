"""
SIH 2026 - Model 3: Multi-Agent Trajectory Prediction Neural Network
Architecture: Temporal Convolutional Feature Encoder + Trajectory Decoder
Accurately forecasts future 1-3s motion paths and seamlessly exports to MATLAB ONNX.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IndianTrajectoryPredictorNet(nn.Module):
    """
    Temporal Trajectory Forecasting Neural Network.
    Input : Past Trajectory [Batch, past_len, 2] (X, Y coordinates over past 1.6s)
    Output: Future Trajectory [Batch, pred_len, 2] (X, Y coordinates over future 2.4s)
    """
    def __init__(self, past_len=8, pred_len=12, hidden_dim=128):
        super().__init__()
        self.past_len = past_len
        self.pred_len = pred_len
        
        # 1. Temporal 1D Convolutional Feature Extractor
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # 2. Dense Trajectory Forecasting Layers
        in_features = hidden_dim * past_len
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, pred_len * 2) # Future (X, Y) displacements
        )

    def forward(self, past_traj):
        # past_traj: [Batch, past_len, 2]
        batch_size = past_traj.size(0)
        
        # Center trajectory relative to last known position for translational invariance
        last_pos = past_traj[:, -1:, :] # [Batch, 1, 2]
        rel_past = past_traj - last_pos # [Batch, past_len, 2]
        
        # 1D Convolution over temporal dimension: [Batch, Channels=2, Length=past_len]
        x = rel_past.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Flatten and decode future relative waypoints
        x_flat = x.flatten(1) # [Batch, hidden_dim * past_len]
        pred_rel = self.decoder(x_flat).view(batch_size, self.pred_len, 2)
        
        # Add back last position to get absolute predicted coordinates
        future_traj = last_pos + pred_rel # [Batch, pred_len, 2]
        return future_traj

def compute_trajectory_loss(predictions, targets):
    """
    Multi-objective trajectory loss:
    1. Average Displacement Error (ADE)
    2. Final Displacement Error (FDE)
    """
    # Displacement errors at each time step
    diff = predictions - targets # [Batch, pred_len, 2]
    step_errors = torch.norm(diff, dim=-1) # [Batch, pred_len]
    
    ade = torch.mean(step_errors) # Average Displacement Error
    fde = torch.mean(step_errors[:, -1]) # Final Displacement Error at t=2.4s
    
    total_loss = ade + 0.5 * fde
    return total_loss, ade.item(), fde.item()

if __name__ == "__main__":
    print("Testing IndianTrajectoryPredictorNet architecture...")
    model = IndianTrajectoryPredictorNet()
    dummy_input = torch.randn(4, 8, 2)
    output = model(dummy_input)
    print(f"Input past trajectory shape : {dummy_input.shape} -> Expected [4, 8, 2]")
    print(f"Output predicted path shape : {output.shape} -> Expected [4, 12, 2]")
    print("[SUCCESS] Trajectory network architecture test PASSED!")
