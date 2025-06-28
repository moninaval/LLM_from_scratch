import torch
import torch.nn as nn
import torch.nn.functional as F
from src.util.debug_util import DebugLogger

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, use_rotary=False, dropout=0.0, debug=False):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rotary = use_rotary

        self.logger = DebugLogger(enabled=debug, prefix="[MultiHeadSelfAttention]")

        # Linear projections for Q, K, V (merged mode)
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape  # Batch, Time, Channels
        self.logger.log(f"Input shape: {x.shape}")

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # Shape: [B, T, 3C]
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: [B, H, T, D]
        self.logger.log(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

        if self.use_rotary:
            self.logger.log("Rotary positional embeddings to be applied (not implemented yet)")
            # TODO: Apply RoPE here (apply_rotary_pos_emb(Q, K))

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T, T]
        self.logger.log(f"Attention scores shape: {attn_scores.shape}")

        if mask is not None:
            self.logger.log("Applying causal mask")
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, T, T]
        attn_probs = self.dropout(attn_probs)
        self.logger.log(f"Attention probabilities shape: {attn_probs.shape}")

        # Weighted sum of values
        context = torch.matmul(attn_probs, V)  # [B, H, T, D]
        context = context.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        out = self.out_proj(context)  # [B, T, C]
        self.logger.log(f"Output shape: {out.shape}")
        return out
