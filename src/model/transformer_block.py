# src/model/transformer_block.py

import torch
import torch.nn as nn
from src.model.attention import MultiHeadSelfAttention
from src.model.feedforward import FeedForward
from src.model.norm import Norm
from src.util.debug_util import DebugLogger

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.0, use_rotary=False, debug=False, norm_type="layernorm"):
        super().__init__()

        self.ln1 = Norm(hidden_size, norm_type)
        self.attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_rotary=use_rotary,
            dropout=dropout,
            debug=debug
        )

        self.ln2 = Norm(hidden_size, norm_type)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout
        )

        self.logger = DebugLogger(enabled=debug, prefix="[TransformerBlock]")

    def forward(self, x, mask=None):
        self.logger.log(f"Input to TransformerBlock: {x.shape}")

        # Attention with residual connection
        x = x + self.attn(self.ln1(x), mask)
        self.logger.log(f"After attention: {x.shape}")

        # Feed-forward with residual connection
        x = x + self.ffn(self.ln2(x))
        self.logger.log(f"After feed-forward: {x.shape}")

        return x
