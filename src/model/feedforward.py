import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.0, ffn_type="gelu"):
        super().__init__()

        self.ffn_type = ffn_type.lower()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        if self.ffn_type == "gelu":
            x = torch.nn.functional.gelu(x)
        elif self.ffn_type == "relu":
            x = torch.nn.functional.relu(x)
        else:
            raise ValueError(f"Unsupported FFN type: {self.ffn_type}")
        x = self.linear2(x)
        return self.dropout(x)

