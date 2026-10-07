"""
SIH 2026 - Model 3 (Multi-Agent Trajectory Prediction)
Dataset Generator for Indian Road Non-Lane Motion Forecasting
Simulates realistic motion for cattle crossings, weaving bikes, pedestrians, and autos.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class IndianRoadTrajectoryDataset(Dataset):
    """
    Synthesizes diverse Indian non-lane traffic trajectory sequences.
    Past Horizon: 8 steps (1.6s @ dt=0.2s)
    Future Horizon: 12 steps (2.4s @ dt=0.2s)
    """
    def __init__(self, num_samples=4000, past_len=8, pred_len=12, dt=0.2):
        self.num_samples = num_samples
        self.past_len = past_len
        self.pred_len = pred_len
        self.dt = dt
        
        np.random.seed(42)
        self.X_data = []
        self.Y_data = []
        
        for _ in range(num_samples):
            agent_type = np.random.choice(['cattle', 'bike', 'pedestrian', 'vehicle'], p=[0.25, 0.35, 0.20, 0.20])
            total_steps = past_len + pred_len
            traj = []
            cx, cy = 0.0, 0.0
            
            if agent_type == 'cattle':
                speed = np.random.uniform(0.6, 1.8)
                heading = np.random.uniform(-np.pi, np.pi)
                for step in range(total_steps):
                    if step >= 6 and np.random.rand() > 0.4:
                        heading += np.random.uniform(-0.5, 0.5)
                    cx += speed * np.cos(heading) * dt + np.random.normal(0, 0.03)
                    cy += speed * np.sin(heading) * dt + np.random.normal(0, 0.03)
                    traj.append([cx, cy])
                    
            elif agent_type == 'bike':
                speed = np.random.uniform(3.5, 7.5)
                base_heading = np.random.uniform(-np.pi, np.pi)
                freq = np.random.uniform(0.5, 1.2)
                for step in range(total_steps):
                    lateral_offset = 0.8 * np.sin(freq * step * dt)
                    cx += speed * np.cos(base_heading) * dt + lateral_offset * (-np.sin(base_heading)) * dt
                    cy += speed * np.sin(base_heading) * dt + lateral_offset * (np.cos(base_heading)) * dt
                    traj.append([cx, cy])
                    
            elif agent_type == 'pedestrian':
                speed = np.random.uniform(0.7, 1.4)
                heading = np.random.uniform(-np.pi, np.pi)
                for _ in range(total_steps):
                    cx += speed * np.cos(heading) * dt + np.random.normal(0, 0.02)
                    cy += speed * np.sin(heading) * dt + np.random.normal(0, 0.02)
                    traj.append([cx, cy])
                    
            else: # vehicle
                speed = np.random.uniform(4.0, 9.0)
                heading = np.random.uniform(-np.pi, np.pi)
                yaw_rate = np.random.choice([0.0, 0.1, -0.1])
                for _ in range(total_steps):
                    cx += speed * np.cos(heading) * dt
                    cy += speed * np.sin(heading) * dt
                    heading += yaw_rate * dt
                    traj.append([cx, cy])
                    
            self.X_data.append(traj[:past_len])
            self.Y_data.append(traj[past_len:])
            
        self.X = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y_data), dtype=torch.float32)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_trajectory_loaders(batch_size=64):
    full_ds = IndianRoadTrajectoryDataset(num_samples=4000)
    train_size = 3200
    val_size = 800
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    ds = IndianRoadTrajectoryDataset(num_samples=10)
    x, y = ds[0]
    print(f"Past Trajectory shape   : {x.shape} (8 steps, 2D coords)")
    print(f"Future Trajectory shape : {y.shape} (12 steps, 2D coords)")
    print("[SUCCESS] Trajectory dataset generator test PASSED!")
