# src/model/embedding.py
import torch
import torch.nn as nn
import os, json
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=2048, use_rotary=False,log_everything=False,log_everything_path="log/current"):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.use_rotary = use_rotary
        self.log_everything = log_everything
        self.log_everything_path=log_everything_path
        if not self.use_rotary:
            self.pos_embed = nn.Embedding(max_position_embeddings, hidden_size)
        else:
            self.pos_embed = None  # rotary will be applied in attention later
    def _log_embeddings_json(self, input_ids, token_embeddings, pos_embeddings, final_embeddings):
        """
        Minimal, newbie-friendly:
      - Always logs sample_index=0, token_index=0
      - Vector = token+pos if learned-pos; token-only if RoPE
      - Overwrites log/current/embedding.json each step
        """
        # Shapes
        B, T, H = final_embeddings.shape
        if B == 0 or T == 0:
            return  # nothing to log

        si, ti = 0, 0  # always first sample & first token
        # (defensive clamp in case batch/seq are smaller)
        if si > B - 1: si = B - 1
        if ti > T - 1: ti = T - 1

        # Pick the vector to show:
        # - RoPE: final == token here (no learned pos added at embedding time)
        # - Learned-pos: final == token + pos
        vec = (final_embeddings[si, ti, :]
           .detach().to("cpu").float().numpy().round(6).tolist())
        token_id = int(input_ids[si, ti].item())

        if getattr(self, "use_rotary", False):
            positioning = {
                "type": "rope",
                "rotary_dim": int(getattr(self, "rotary_dim", 0)) or None
            }
            note = "Vector shows token embedding only; positional phase (RoPE) is applied to Q,K later in attention."
            formula = "E[i,t] = W_token[ids[i,t]]  (RoPE applied later in attention)"
        else:
            positioning = {"type": "learned"}
            note = "Vector includes token + learned positional embedding at this position."
            formula = "E[i,t] = W_token[ids[i,t]] + W_pos[t]"

        payload = {
            "sample": {
            "shape": [int(B), int(T), int(H)],
            "sample_index": si,
            "token_index": ti,
            "token_id": token_id,
            "vector_len": int(H),
            "vector": vec
            },
            "positioning": positioning,
            "note": note,
            "formula": formula
        }

        out_dir = self.log_everything_path
        os.makedirs(out_dir, exist_ok=True)
        tmp = os.path.join(out_dir, "embedding.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(out_dir, "embedding.json"))
        
    def forward(self, input_ids):
        # input_ids: (B, T)
        token_embeddings = self.token_embed(input_ids)

        if self.use_rotary:
            # RoPE: no learned positional embedding here
            pos_embeddings = None
            final_embeddings = token_embeddings
        else:
            # Learned positional embeddings
            B, T = input_ids.shape
            positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            pos_embeddings = self.pos_embed(positions)           # (B, T, H)
            final_embeddings = token_embeddings + pos_embeddings  # (B, T, H)

        # Logging (single call; your _log_embeddings_json handles rotary vs learned)
        if getattr(self, "log_everything", False):
            try:
                self._log_embeddings_json(input_ids, token_embeddings, pos_embeddings, final_embeddings)
            except Exception as e:
                print(f"[embedding log error] {e}")

        return final_embeddings
