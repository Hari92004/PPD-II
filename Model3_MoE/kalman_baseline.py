"""
====================================================================================================
Model 3 Baseline 1: Constant-Velocity Kalman Filter Trajectory Predictor
====================================================================================================
- State vector: x = [x, y, vx, vy]^T
- Time step: dt = 0.2 s (5 Hz)
- Filters past 8 observed steps (1.6s) to estimate optimal position & velocity.
- Predicts future 12 steps (2.4s) via constant velocity state extrapolation.
- Evaluates ADE, FDE, and Miss Rate (FDE > 2.0m) overall and per-class.
====================================================================================================
"""

import numpy as np
import torch
from collections import defaultdict
from moe_dataset import CLASSES, build_moe_datasets

class ConstantVelocityKalmanFilter:
    """
    2D Constant-Velocity Kalman Filter for Trajectory Forecasting.
    """
    def __init__(self, dt=0.2, process_noise_std=0.5, measurement_noise_std=0.2):
        self.dt = dt
        # State transition matrix F
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Measurement matrix H
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        # Process noise covariance Q (continuous white noise acceleration model)
        q = process_noise_std ** 2
        dt2 = (dt ** 2) / 2.0
        dt3 = (dt ** 3) / 3.0
        self.Q = q * np.array([
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2],
            [dt2, 0.0, dt,  0.0],
            [0.0, dt2, 0.0, dt]
        ], dtype=np.float32)

        # Measurement noise covariance R
        r = measurement_noise_std ** 2
        self.R = r * np.eye(2, dtype=np.float32)

    def predict_trajectory(self, past_positions, pred_len=12):
        """
        past_positions: [8, 2] numpy array
        Returns:
          future_positions: [12, 2] numpy array
        """
        # 1. Initialize state with first observation and finite-difference velocity
        x0, y0 = past_positions[0]
        vx0 = (past_positions[1, 0] - past_positions[0, 0]) / self.dt if len(past_positions) > 1 else 0.0
        vy0 = (past_positions[1, 1] - past_positions[0, 1]) / self.dt if len(past_positions) > 1 else 0.0

        x = np.array([x0, y0, vx0, vy0], dtype=np.float32)
        P = np.diag([1.0, 1.0, 5.0, 5.0]).astype(np.float32) # Initial state covariance
        I = np.eye(4, dtype=np.float32)

        # 2. Filter through remaining observed past steps (steps 1 to 7)
        for t in range(1, len(past_positions)):
            # Predict
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q

            # Update with measurement z = [x_meas, y_meas]
            z = past_positions[t]
            y = z - (self.H @ x) # Innovation
            S = self.H @ P @ self.H.T + self.R # Innovation covariance
            K = P @ self.H.T @ np.linalg.inv(S) # Kalman gain
            x = x + K @ y
            P = (I - K @ self.H) @ P

        # 3. Forecast future pred_len steps by recursive state transition
        future_pred = []
        curr_x = x.copy()
        for _ in range(pred_len):
            curr_x = self.F @ curr_x
            future_pred.append([curr_x[0], curr_x[1]])

        return np.array(future_pred, dtype=np.float32)


def evaluate_kalman_baseline(val_dataset):
    """
    Evaluates Constant Velocity Kalman Filter on the validation dataset.
    Returns:
      overall_metrics: dict with ADE, FDE, MissRate
      per_class_metrics: dict mapping class name to metrics
    """
    kf = ConstantVelocityKalmanFilter(dt=0.2)
    per_class_errors = defaultdict(lambda: {'ade': [], 'fde': []})
    all_ade = []
    all_fde = []

    print("\n[EVALUATION] Running Constant-Velocity Kalman Filter on Validation Set...")
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        past = sample['past_pos'].numpy()     # [8, 2]
        future_gt = sample['future_pos'].numpy() # [12, 2]
        cid = int(sample['class_id'])
        cname = CLASSES[cid]

        # Forecast
        pred_future = kf.predict_trajectory(past, pred_len=12)

        # Compute ADE & FDE
        diff = pred_future - future_gt
        step_errors = np.linalg.norm(diff, axis=-1) # [12]
        ade = float(np.mean(step_errors))
        fde = float(step_errors[-1])

        all_ade.append(ade)
        all_fde.append(fde)
        per_class_errors[cname]['ade'].append(ade)
        per_class_errors[cname]['fde'].append(fde)

    overall_ade = np.mean(all_ade)
    overall_fde = np.mean(all_fde)
    overall_miss_rate = np.mean(np.array(all_fde) > 2.0) * 100.0

    per_class_summary = {}
    for cname in CLASSES:
        if cname in per_class_errors and len(per_class_errors[cname]['ade']) > 0:
            c_ades = per_class_errors[cname]['ade']
            c_fdes = per_class_errors[cname]['fde']
            c_miss = np.mean(np.array(c_fdes) > 2.0) * 100.0
            per_class_summary[cname] = {
                'ade': np.mean(c_ades),
                'fde': np.mean(c_fdes),
                'miss_rate': c_miss,
                'count': len(c_ades)
            }

    print("\n" + "=" * 75)
    print("   BASELINE 1: CONSTANT-VELOCITY KALMAN FILTER RESULTS")
    print("=" * 75)
    print(f"OVERALL : ADE = {overall_ade:.3f} m | FDE = {overall_fde:.3f} m | Miss Rate (>2m) = {overall_miss_rate:.2f}%")
    print("-" * 75)
    print(f"{'Class':<15} | {'Samples':<8} | {'ADE (m)':<10} | {'FDE (m)':<10} | {'Miss Rate (%)':<15}")
    print("-" * 75)
    for cname in CLASSES:
        if cname in per_class_summary:
            m = per_class_summary[cname]
            print(f"{cname:<15} | {m['count']:<8} | {m['ade']:<10.3f} | {m['fde']:<10.3f} | {m['miss_rate']:<15.2f}%")
    print("=" * 75)

    return {
        'overall': {'ade': overall_ade, 'fde': overall_fde, 'miss_rate': overall_miss_rate},
        'per_class': per_class_summary
    }


if __name__ == "__main__":
    _, val_dataset = build_moe_datasets()
    evaluate_kalman_baseline(val_dataset)

