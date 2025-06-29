import torch.nn as nn
from src.model.transformer_block import TransformerBlock
from src.model.norm import Norm

class TransformerStack(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, dropout=0.0, use_rotary=False,
                 rotary_dim=None, norm_type="layernorm", ffn_type="gelu", qkv_proj="fused", debug=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                use_rotary=use_rotary,
                rotary_dim=rotary_dim,
                norm_type=norm_type,
                ffn_type=ffn_type,
                qkv_proj=qkv_proj,
                debug=debug
            ) for _ in range(num_layers)
        ])
        self.final_norm = Norm(hidden_size, norm_type)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x)
