import torch.nn as nn
from src.model.attention import MultiHeadSelfAttention
from src.model.feedforward import FeedForward
from src.model.norm import Norm

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.0, use_rotary=False,
                 rotary_dim=None, norm_type="layernorm", ffn_type="gelu", qkv_proj="fused", debug=False):
        super().__init__()

        self.ln1 = Norm(hidden_size, norm_type)
        self.attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            rotary_dim=rotary_dim,
            qkv_proj=qkv_proj,
            debug=debug
        )
        self.ln2 = Norm(hidden_size, norm_type)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            ffn_type=ffn_type
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x