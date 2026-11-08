"""
====================================================================================================
Model 3: Mixture-of-Experts (MoE) GRU Trajectory Predictor & Single-GRU Baseline
====================================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleGRUExpert(nn.Module):
    """
    Individual Expert Module: 2-layer GRU Encoder + GRU Decoder.
    Encodes 8 past steps and decodes 12 future step offsets (dx, dy).
    """
    def __init__(self, in_dim=80, hidden_dim=128, pred_len=12):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # 2-layer GRU Encoder
        self.encoder = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # GRU Decoder Cell
        self.decoder_cell = nn.GRUCell(
            input_size=2,
            hidden_size=hidden_dim
        )

        # Step offset projection
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x_seq, last_vel):
        B = x_seq.size(0)
        _, h_n = self.encoder(x_seq)
        h_dec = h_n[-1]

        curr_offset = last_vel
        predicted_offsets = []

        for _ in range(self.pred_len):
            h_dec = self.decoder_cell(curr_offset, h_dec)
            curr_offset = self.offset_head(h_dec)
            predicted_offsets.append(curr_offset.unsqueeze(1))

        offsets = torch.cat(predicted_offsets, dim=1)
        return offsets


class MoEGRUTrajectoryNet(nn.Module):
    def __init__(self, num_classes=7, class_emb_dim=16, hidden_dim=128,
                 past_len=8, pred_len=12, num_experts=4, dt=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.hidden_dim = hidden_dim
        self.past_len = past_len
        self.pred_len = pred_len
        self.num_experts = num_experts
        self.dt = dt

        self.class_embedding = nn.Embedding(num_classes, class_emb_dim)

        self.kinematic_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        total_in_dim = 64 + class_emb_dim

        self.experts = nn.ModuleList([
            SingleGRUExpert(in_dim=total_in_dim, hidden_dim=hidden_dim, pred_len=pred_len)
            for _ in range(num_experts)
        ])

        if num_experts > 1:
            gating_in_dim = class_emb_dim + 2 + 64
            self.gating_net = nn.Sequential(
                nn.Linear(gating_in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_experts)
            )

    def forward(self, past_positions, class_id):
        B, T, _ = past_positions.shape
        if class_id.dim() > 1:
            class_id = class_id.squeeze(-1)

        origin = past_positions[:, -1:, :]
        rel_pos = past_positions - origin

        v0 = rel_pos[:, 1:2, :] - rel_pos[:, 0:1, :]
        v_rest = rel_pos[:, 1:, :] - rel_pos[:, :-1, :]
        v = torch.cat([v0, v_rest], dim=1) / self.dt

        kinematics = torch.cat([rel_pos, v], dim=-1)
        kin_feats = self.kinematic_mlp(kinematics)

        class_emb = self.class_embedding(class_id)
        class_emb_seq = class_emb.unsqueeze(1).expand(-1, T, -1)

        expert_in = torch.cat([kin_feats, class_emb_seq], dim=-1)
        last_vel = v[:, -1, :] * self.dt

        if self.num_experts > 1:
            hist_summary = kin_feats.mean(dim=1)
            gating_input = torch.cat([class_emb, last_vel, hist_summary], dim=-1)
            gate_logits = self.gating_net(gating_input)
            gates = F.softmax(gate_logits, dim=-1)
        else:
            gates = torch.ones((B, 1), device=past_positions.device, dtype=torch.float32)

        expert_outputs = torch.stack([
            expert(expert_in, last_vel) for expert in self.experts
        ], dim=0)

        gates_expanded = gates.t().unsqueeze(-1).unsqueeze(-1)
        fused_offsets = (expert_outputs * gates_expanded).sum(dim=0)

        pred_rel_positions = torch.cumsum(fused_offsets, dim=1)
        future_positions = pred_rel_positions + origin

        return future_positions, gates


def compute_moe_loss(pred_pos, target_pos, gates, class_id=None, class_weights=None, lambda_balance=0.1):
    """
    Multi-objective loss function with per-class weighting support.
    """
    # [B, 12, 2]
    step_loss = F.smooth_l1_loss(pred_pos, target_pos, beta=0.1, reduction='none')
    sample_pos_loss = step_loss.mean(dim=[1, 2]) # [B]

    if class_id is not None and class_weights is not None:
        w = class_weights[class_id] # [B]
        pos_loss = (sample_pos_loss * w).mean()
    else:
        pos_loss = sample_pos_loss.mean()

    num_experts = gates.size(-1)
    if num_experts > 1:
        mean_gate_usage = gates.mean(dim=0)
        balance_loss = num_experts * torch.sum(mean_gate_usage ** 2) - 1.0
    else:
        balance_loss = torch.tensor(0.0, device=pred_pos.device)

    total_loss = pos_loss + (lambda_balance * balance_loss)
    return total_loss, pos_loss, balance_loss
