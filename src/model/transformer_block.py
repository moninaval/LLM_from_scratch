# src/model/transformer_block.py  (or wherever your block class is)

import torch.nn as nn
from src.model.attention import MultiHeadSelfAttention
from src.model.feedforward import FeedForward
from src.model.norm import Norm

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size,
                 dropout=0.0, use_rotary=False, rotary_dim=None,
                 norm_type="layernorm", ffn_type="gelu",
                 qkv_proj="fused", debug=False, log_everything=True,log_everything_path="log/current"):
        super().__init__()
        self.ln1  = Norm(hidden_size, norm_type)
        self.attn = MultiHeadSelfAttention(
            hidden_size, num_heads, dropout=dropout,
            use_rotary=use_rotary, rotary_dim=rotary_dim,
            qkv_proj=qkv_proj, debug=debug, log_everything=log_everything
        )
        self.ln2  = Norm(hidden_size, norm_type)
        self.ffn  = FeedForward(hidden_size, intermediate_size, dropout=dropout, ffn_type=ffn_type)

        # logging knobs (propagate same ones you already use)
        self.log_everything = log_everything
        self.log_dir        = log_everything_path
      #  self.layer_index    = -1           # set by TransformerStack when building

    def forward(self, x, mask=None):
        # keep stepwise tensors so we can log exactly what flowed
        x_in   = x
        x_ln1  = self.ln1(x_in)
        a_out  = self.attn(x_ln1, mask)           # attention already logs heads/merge
        y1     = x_in + a_out                     # residual after attn

        y1_ln2 = self.ln2(y1)
        f_out  = self.ffn(y1_ln2)                 # FFN output
        y2     = y1 + f_out                       # residual after ffn (block output)
        if getattr(self, "log_everything", True):
            try:
                self._log_block_trace(x_in, x_ln1, a_out, y1, y1_ln2, f_out, y2,self.attn.layer_index)
            except Exception as e:
                print(f"[block log error] {e}")

        return y2
    
    

    def _log_block_trace(self, x_in, ln1_out, attn_out, y1, ln2_out, ffn_out, y2,layer_index):
        import os, json
        out_dir = getattr(self, "log_dir", "log/current")
        os.makedirs(out_dir, exist_ok=True)
        def _to_list6(t):
    # Safe: detaches, moves to CPU, uses NumPy to round, returns JSON-friendly list
            return t.detach().cpu().numpy().round(6).tolist()
        B, T, H = x_in.shape
        si = max(0, min(int(getattr(self, "log_sample_index", 0)), B - 1))
        ti = max(0, min(int(getattr(self, "log_token_index", 0)), T - 1))
        payload = {
            "layer_index": layer_index,
            "sample_index": int(si),
            "token_index": int(ti),
            "shapes": {
                "x_in": list(x_in.shape),
                "ln1_out": list(ln1_out.shape),
                "attn_out": list(attn_out.shape),
                "y1": list(y1.shape),
                "ln2_out": list(ln2_out.shape),
                "ffn_out": list(ffn_out.shape),
                "y2": list(y2.shape),
            },
            "flow": [
                {"stage": "x_in",     "vec": _to_list6(x_in[si, ti, :])},
                {"stage": "ln1(x)",   "vec": _to_list6(ln1_out[si, ti, :])},
                {"stage": "attn_out", "vec": _to_list6(attn_out[si, ti, :])},
                {"stage": "residual1","vec": _to_list6(y1[si, ti, :])},
                {"stage": "ln2(y1)",  "vec": _to_list6(ln2_out[si, ti, :])},
                {"stage": "ffn_out",  "vec": _to_list6(ffn_out[si, ti, :])},
                {"stage": "residual2","vec": _to_list6(y2[si, ti, :])},
            ],
            "formula": [
                "u = LN1(x);  a = MHA(u);         y1 = x + a",
                "v = LN2(y1); f = FFN(v);         y2 = y1 + f"
            ],
        }

        # (optional) attach attention merge snapshot from the attention module if present
       
        print("NAVAL creating FFN.json")
        fname = f"decoder_layer_{layer_index}_FFN.json"
        path = os.path.join(out_dir, fname)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)

