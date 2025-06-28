# src/model/transformer_stack.py

import torch
import torch.nn as nn
from src.model.transformer_block import TransformerBlock
from src.util.debug_util import DebugLogger

class TransformerStack(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, dropout=0.0, use_rotary=False, debug=False, norm_type="layernorm"):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                use_rotary=use_rotary,
                debug=debug,
                norm_type=norm_type
            ) for _ in range(num_layers)
        ])

        self.logger = DebugLogger(enabled=debug, prefix="[TransformerStack]")

    def forward(self, x, mask=None):
        self.logger.log(f"Input to TransformerStack: {x.shape}")
        for i, block in enumerate(self.blocks):
            self.logger.log(f"\n>>> Block {i}")
            x = block(x, mask)
        self.logger.log(f"Output of TransformerStack: {x.shape}")
        return x
