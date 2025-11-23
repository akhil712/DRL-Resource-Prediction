# src/ml/fedbi_gru.py
import torch
import torch.nn as nn

class BiGRUModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=20, num_layers=2, out_dim=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, out_dim)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc(out)
        return out
