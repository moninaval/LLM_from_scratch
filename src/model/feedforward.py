# src/model/feedforward.py

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.0, activation="gelu"):
        super().__init__()
        self.activation = activation.lower()

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.activation == "gelu":
            x = torch.nn.functional.gelu(self.linear1(x))
        elif self.activation == "relu":
            x = torch.nn.functional.relu(self.linear1(x))
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        x = self.linear2(x)
        return self.dropout(x)
