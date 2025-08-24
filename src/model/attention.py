import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, use_rotary=False, rotary_dim=None, qkv_proj="fused", debug=False,log_everything=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rotary = use_rotary
        self.rotary_dim = rotary_dim or self.head_dim
        self.qkv_proj = qkv_proj
        self.debug = debug
        self.log_everything=log_everything

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
    def _log_heads_trace(self,q, k, v,scores_raw, scores_masked,weights_soft, weights_post,context_heads,concat=None, wo_out=None):
        # q,k,v: [B,h,T,d]
        # scores_* / weights_*: [B,h,T,T]
        # context_heads: [B,h,T,d]
        import os, json, torch

        out_dir = getattr(self, "log_dir", "log/current")
        os.makedirs(out_dir, exist_ok=True)

        si   = int(getattr(self, "log_sample_index", 0))   # which batch row to log
        ti   = int(getattr(self, "log_token_index", 0))    # which token position to log
        topk = int(getattr(self, "log_topk", 5))

        B, h, T, d = q.size()
        si   = max(0, min(si, B - 1))
        ti   = max(0, min(ti, T - 1))
        topk = max(1, min(topk, T))

        heads = []
        for head in range(self.num_heads):
            q_vec  = q[si, head, ti, :].detach().cpu().float()
            k_mat  = k[si, head, :, :].detach().cpu().float()
            v_mat  = v[si, head, :, :].detach().cpu().float()
            s_raw  = scores_raw[si, head, ti, :].detach().cpu().float()
            s_mask = scores_masked[si, head, ti, :].detach().cpu().float()
            w_soft = weights_soft[si, head, ti, :].detach().cpu().float()
            w_post = weights_post[si, head, ti, :].detach().cpu().float()
            c_head = context_heads[si, head, ti, :].detach().cpu().float()

            trace = []
            def step(name, dct):
                trace.append({"stage_idx": len(trace), "stage": name, **dct})

            # 0) Q after proj (and RoPE if enabled)
            step("q_after_proj" + ("_rope" if self.use_rotary else ""), {
                "q": q_vec.numpy().round(6).tolist()
            })
            # NEW: log all keys for this head
            step("k_after_proj" + ("_rope" if self.use_rotary else ""), {
                "k": k_mat.numpy().round(6).tolist()
            })
            # NEW: log all values for this head
            step("v_after_proj", {
                "v": v_mat.numpy().round(6).tolist()
            })
            # 1) raw scores
            step("scores_raw", {
                "scores": s_raw.numpy().round(6).tolist()
            })

            # 2) masked scores
            step("scores_masked", {
                "scores": s_mask.numpy().round(6).tolist()
            })

            # 3) softmax weights (pre-dropout)
            step("weights_softmax", {
                "weights": w_soft.numpy().round(6).tolist(),
                "sum": float(w_soft.sum().item())
            })

            # 4) post-dropout weights (used for context)
            drop_applied = bool(self.training and getattr(self.dropout, "p", 0.0) > 0.0)
            step("weights_after_dropout", {
                "weights": w_post.numpy().round(6).tolist(),
                "sum": float(w_post.sum().item()),
                "dropout_applied": drop_applied
            })

            # 5) top-k by post-dropout weights
            _, idx = torch.topk(w_post, k=topk)
            top_entries = []
            for j in idx:
                jj = int(j)
                top_entries.append({
                    "pos": jj,
                    "score_masked": float(s_mask[jj]),
                    "weight": float(w_post[jj]),
                    "k": k_mat[jj, :].numpy().round(6).tolist(),
                    "v": v_mat[jj, :].numpy().round(6).tolist()
                })
            step("topk_after_dropout", {"k": topk, "topk": top_entries})

            # 6) per-head context
            step("context_head", {"context": c_head.numpy().round(6).tolist()})

            heads.append({"head": head, "head_dim": int(d), "trace": trace})

        # ---- attach compact flow snapshot for the Output panel ----
        # pick (si, ti) consistent with above
        per_head_ctx = [ context_heads[si, hh, ti, :].detach().cpu().float().tolist()
                        for hh in range(self.num_heads) ]
        concat_vec = None
        wo_vec = None
        if concat is not None:
            concat_vec = concat[si, ti, :].detach().cpu().float().tolist()
        if wo_out is not None:
            wo_vec = wo_out[si, ti, :].detach().cpu().float().tolist()

        self._last_flow_attn = {
            "layer_index": int(getattr(self, "layer_index", -1)),
            "sample_index": int(si),
            "token_index": int(ti),
            "num_heads": int(self.num_heads),
            "head_dim": int(self.head_dim),
            "per_head_context": per_head_ctx,               # list of H vectors (d_head each)
            "concat_vector": concat_vec,                    # (C,) if provided
            "wo_output": wo_vec,                            # (C,) if provided
            "formula": [
                "Context_h = softmax(Q_h K_h^T / √d_head) · V_h",
                "Concat = [Context_0 || ... || Context_{H-1}]",
                "AttnOut = Concat · W_O^T"
            ]
        }

        # ---- write the per-layer, per-head trace file (unchanged + extras) ----
      
        li = int(getattr(self, "layer_index", 0))
        print("NAVAL from attention",li)
        payload = {
            "layer_index": li,
            "sample_index": si,
            "token_index": ti,
            "num_heads": int(self.num_heads),
            "head_dim": int(self.head_dim),
            "rope": bool(self.use_rotary),
            "rope_note": "Q,K are AFTER RoPE if enabled",
            "formula": [
                "Q = LN(x)·WQ, K = LN(x)·WK, V = LN(x)·WV",
                "scores_raw = (QK^T)/√d",
                "scores_masked = scores_raw + masks",
                "weights_softmax = softmax(scores_masked)",
                "weights_after_dropout = dropout(weights_softmax)",
                "context_h(t) = Σ_j weights_after_dropout[j] · V[j]",
                "Concat = [context_h]_h ; AttnOut = Concat · W_O^T"
            ],
            "merge": {
                "concat_vector": concat_vec,   # may be None if not passed
                "wo_output": wo_vec            # may be None if not passed
            },
            "heads": heads
        }

        fname = f"decoder_layer_{li}.json"
        tmp = os.path.join(out_dir, fname + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(out_dir, fname))

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # QKV
        if self.qkv_proj == "fused":
            qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]          # [B,h,T,d]
        else:
            q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE (if enabled)
        if self.use_rotary:
            q, k = self.apply_rotary(q, k)

        # ---- STEP 1: raw scores
        scores_raw = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # [B,h,T,T]

        # ---- STEP 2: masking
        if mask is None:
            scores_masked = scores_raw
        else:
            scores_masked = scores_raw.masked_fill(mask == 0, float('-inf'))

        # ---- STEP 3: softmax (pre-dropout)
        weights_soft = torch.softmax(scores_masked, dim=-1)                             # [B,h,T,T]

        # ---- STEP 4: dropout (post-softmax)
        weights_post = self.dropout(weights_soft)                                       # [B,h,T,T]

        # ---- STEP 5: per-head context, then merge heads
        context_heads = torch.matmul(weights_post, v)                                   # [B,h,T,d]
        context = context_heads.transpose(1, 2).contiguous().view(B, T, C)              # [B,T,C]
        concat = context_heads.transpose(1, 2).contiguous().view(B, T, C)      # (B,T,C)
        wo_out = self.out_proj(concat)                                     # (B,T,C)


        # logging in the SAME order as computed
        if getattr(self, "log_everything", True):
            try:
                self._log_heads_trace(q, k, v, scores_raw, scores_masked, weights_soft, weights_post, context_heads,concat=concat,                    # (B,T,C)  ← new
            wo_out=wo_out )
            except Exception as e:
                print(f"[attn log error] {e}")

        return self.out_proj(context)


    def apply_rotary(self, q, k):
        return q, k  # placeholder (rotary embeddings not implemented yet)
