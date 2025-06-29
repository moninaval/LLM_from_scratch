import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_rotary=False, rotary_dim=None, qkv_proj="fused", debug=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rotary = use_rotary
        self.rotary_dim = rotary_dim or self.head_dim
        self.qkv_proj = qkv_proj
        self.debug = debug

        if qkv_proj == "fused":
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        elif qkv_proj == "split":
            self.q = nn.Linear(hidden_size, hidden_size)
            self.k = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, hidden_size)
        else:
            raise ValueError(f"Unsupported qkv_proj type: {qkv_proj}")

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        if self.qkv_proj == "fused":
            qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary:
            q, k = self.apply_rotary(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(context)

    def apply_rotary(self, q, k):
        return q, k  # placeholder (rotary embeddings not implemented yet)