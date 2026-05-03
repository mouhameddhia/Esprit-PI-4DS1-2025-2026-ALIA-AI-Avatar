from __future__ import annotations

import torch
from torch import nn


class StressLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        dense_dim: int = 128,
        num_classes: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.lstm(x)
        # Use last time step representation for sequence-level stress prediction.
        last = seq_out[:, -1, :]
        return self.classifier(last)
