import torch.nn as nn
from src.model.transformer_block import TransformerBlock
from src.model.norm import Norm

class TransformerStack(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, intermediate_size, dropout=0.0, use_rotary=False,
                 rotary_dim=None, norm_type="layernorm", ffn_type="gelu", qkv_proj="fused", debug=False,log_everything=False,log_everything_path="log/current"):
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
                debug=debug,
                log_everything=log_everything,
                log_everything_path=log_everything_path
            ) for _ in range(num_layers)
        ])
        log_dir = getattr(self, "log_dir", "log/current")
        for i, blk in enumerate(self.blocks):
            attn = getattr(blk, "attn", None) or getattr(blk, "self_attn", None)  # be robust to naming
            if attn is not None:
                attn.layer_index     = i
                attn.log_dir         = log_everything_path
                attn.log_everything  = log_everything
            # optional knobs so you can change which token/top-k to inspect without editing the class
                attn.log_sample_index = getattr(self, "log_sample_index", 0)
                attn.log_token_index  = getattr(self, "log_token_index", 0)
                attn.log_topk         = getattr(self, "log_topk", 5)
        self.final_norm = Norm(hidden_size, norm_type)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x)
