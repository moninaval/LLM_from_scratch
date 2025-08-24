# src/model/llm_model.py

import torch
import torch.nn as nn
from src.model.embedding import TokenEmbedding
from src.model.transformer_stack import TransformerStack
from src.model.norm import Norm
from src.util.debug_util import DebugLogger
import os
import json

class LLMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size,
                 max_position_embeddings=2048, dropout=0.0, use_rotary=False, rotary_dim=None,
                 norm_type="layernorm", ffn_type="gelu", qkv_proj="fused", tie_weights=False, debug=False,log_everything=False,log_chunk=256,log_everything_path="log/current"):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            use_rotary=use_rotary,log_everything=log_everything,log_everything_path=log_everything_path
        )

        self.transformer = TransformerStack(
            num_layers=num_layers,
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
        )

        self.final_norm = Norm(hidden_size, norm_type)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.token_embed.weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = DebugLogger(enabled=debug, prefix="[LLMModel]")
        self.log_everything = log_everything
        self.log_dir=log_everything_path
    # ----------------------------------------------------
    def _log_output_panel(self, x_pre_norm, x_norm, logits, topk=5):
        """
        Saves a compact step-by-step snapshot for the Output panel:
        - per-head contexts/concat/Wo from the last attention block (if present)
        - transformer out (pre final LN)
        - final LN vector
        - logits top-k and probs
        """
        out_dir = getattr(self, "log_dir", "log/current")
        os.makedirs(out_dir, exist_ok=True)

        # which sample/token to show (same knobs you used elsewhere)
        si = int(getattr(self, "log_sample_index", 0))
        ti = int(getattr(self, "log_token_index", 0))
        topk = int(getattr(self, "log_topk", topk))

        B, T, C = x_norm.shape
        si = max(0, min(si, B - 1))
        ti = max(0, min(ti, T - 1))
        topk = max(1, min(topk, logits.size(-1)))

        # from last layer attention flow (if your attention logger set it)
        attn_flow = None
        try:
            last_block = self.transformer.blocks[-1]
            attn_flow = getattr(last_block.attn, "_last_flow_attn", None)
        except Exception:
            attn_flow = None

        # stage vectors
        vec_transformer_out = x_pre_norm[si, ti, :].detach().cpu().float()
        vec_final_norm      = x_norm[si, ti, :].detach().cpu().float()
        vec_logits          = logits[si, ti, :].detach().cpu().float()

        # softmax top-k (don’t dump full vocab)
        probs = torch.softmax(vec_logits, dim=-1)
        vk, ik = torch.topk(probs, k=topk)
        topk_list = [{"id": int(i.item()), "prob": float(v.item())} for v, i in zip(vk, ik)]

        payload = {
            "sample_index": si,
            "token_index": ti,
            "hidden_size": int(C),
            "vocab_size": int(logits.size(-1)),
            "from_attention": attn_flow,  # includes per_head_context, concat_vector, wo_output if available
            "stages": [
                {"name": "transformer_out (pre_final_norm)", "vector": vec_transformer_out.tolist()},
                {"name": "final_norm",                       "vector": vec_final_norm.tolist()},
                {"name": "lm_head_logits_topk",              "topk": topk_list}
            ],
            "formula": [
                "Concat = [context_h]_h ; AttnOut = Concat · W_O^T",
                "x_L = residual stacks over L decoder blocks",
                "ŷ = LN(x_L)",
                "logits = ŷ · W^T  (tied or separate)",
                "p = softmax(logits)"
            ]
        }

        # atomic write
        tmp = os.path.join(out_dir, "output.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(out_dir, "output.json"))
    def forward(self, input_ids, mask=None):
        self.logger.log(f"Input IDs shape: {input_ids.shape}")
        x = self.token_embedding(input_ids)
        self.logger.log(f"After TokenEmbedding: {x.shape}")
        x = self.transformer(x, mask)
        self.logger.log(f"After TransformerStack: {x.shape}")
        x_pre_norm = x
        x = self.final_norm(x)
        self.logger.log(f"After Final Norm: {x.shape}")

        logits = self.lm_head(x)
        self.logger.log(f"Logits shape: {logits.shape}")
        if getattr(self, "log_everything", True):
            print("NAVAL")
            try:
                self._log_output_panel(x_pre_norm, x, logits)
            except Exception as e:
                self.logger.log(f"[output log error] {e}")
        return logits
