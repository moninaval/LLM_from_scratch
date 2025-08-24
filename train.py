import argparse
import json
import math
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from inspect import signature

# --------------------------- Optional dataset helpers ---------------------------

class BinDataset(Dataset):
    """Minimal BinDataset (uint16, shape: (N, seq_len)) — only used if you pass a .bin"""
    def __init__(self, bin_path: str, meta_path: str = None):
        import numpy as np
        self.bin_path = bin_path
        self.meta_path = meta_path or (bin_path + ".meta.json")
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        self.seq_len = int(meta["seq_len"])
        self.n_sequences = int(meta["n_sequences"])
        self._arr = np.memmap(self.bin_path, dtype=np.uint16, mode="r", shape=(self.n_sequences, self.seq_len))

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        import numpy as np
        x = self._arr[idx].astype(np.int64)  # LongTensor ids
        return torch.from_numpy(x)

try:
    from src.data.text_dataset import TextDataset
except Exception:
    class TextDataset(Dataset):
        """Minimal JSONL loader: each line is {"input_ids": [int, ...]}"""
        def __init__(self, file_path: str, key: str = "input_ids"):
            self.path = file_path
            self.key = key
            self.index = []
            with open(self.path, "r", encoding="utf-8") as f:
                off = 0
                for line in f:
                    self.index.append(off)
                    off += len(line.encode("utf-8"))

        def __len__(self):
            return len(self.index)

        def __getitem__(self, idx):
            with open(self.path, "r", encoding="utf-8") as f:
                f.seek(self.index[idx])
                obj = json.loads(f.readline())
            arr = obj.get("input_ids") or obj.get("ids")
            return torch.tensor(arr, dtype=torch.long)

# --------------------------- Model import ---------------------------

from src.model.llm_model import LLMModel
try:
    from src.tokenizer.tokenizer_manager import TokenizerManager
except Exception:
    TokenizerManager = None

# --------------------------- Resumable per-epoch sampler ---------------------------

class ResumableEpochBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size: int, batch_size: int, *, seed: int = 42,
                 shuffle: bool = True, drop_last: bool = True):
        self.N = int(dataset_size)
        self.B = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.batch_pos = 0
        self._build_epoch()

    def _build_epoch(self):
        order = np.arange(self.N)
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(order)
        if self.drop_last:
            usable = (self.N // self.B) * self.B
            order = order[:usable]
        self.num_batches = 0 if len(order) == 0 else len(order) // self.B
        if self.num_batches == 0:
            self.order = np.empty((0, self.B), dtype=int)
        else:
            self.order = order.reshape(-1, self.B)
        self.batch_pos = min(self.batch_pos, self.num_batches)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.batch_pos = 0
        self._build_epoch()

    def state_dict(self):
        return {"epoch": int(self.epoch), "batch_pos": int(self.batch_pos)}

    def load_state_dict(self, state):
        self.epoch = int(state.get("epoch", 0))
        self.batch_pos = int(state.get("batch_pos", 0))
        self._build_epoch()

    def __iter__(self):
        while self.batch_pos < self.num_batches:
            batch = self.order[self.batch_pos].tolist()
            self.batch_pos += 1
            yield batch

    def __len__(self):
        return self.num_batches

# --------------------------- Checkpoint I/O ---------------------------

def save_ckpt(path, model, optimizer, scaler, global_step, batch_sampler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": int(global_step),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler_state": batch_sampler.state_dict() if batch_sampler is not None else None,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_ckpt(path, model, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        try: scaler.load_state_dict(ckpt["scaler"])
        except Exception: pass
    if ckpt.get("rng_state") is not None: torch.set_rng_state(ckpt["rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    global_step = int(ckpt.get("global_step", 0))
    sampler_state = ckpt.get("sampler_state", None)
    return global_step, sampler_state

# --------------------------- LR schedule ---------------------------

def make_lr_fn(cfg_lr, total_step_budget):
    base_lr = float(cfg_lr.get("lr", 3e-5))
    schedule = str(cfg_lr.get("schedule", "cosine")).lower()
    min_lr_ratio = float(cfg_lr.get("min_lr_ratio", 0.1))
    warmup_steps_cfg = cfg_lr.get("warmup_steps", None)
    warmup_ratio = float(cfg_lr.get("warmup_ratio", 0.02))
    warmup_steps = int(warmup_steps_cfg) if warmup_steps_cfg is not None else max(1, int(warmup_ratio * total_step_budget))

    def lr_at(step: int) -> float:
        if schedule == "constant":
            return base_lr
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        if schedule == "linear":
            t = (step - warmup_steps) / max(1, total_step_budget - warmup_steps)
            return base_lr * (1.0 - t)
        # cosine
        t = (step - warmup_steps) / max(1, total_step_budget - warmup_steps)
        return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))
    return lr_at

# --------------------------- Eval ---------------------------

@torch.no_grad()
def evaluate_model(model, val_loader, device, amp_enabled: bool):
    if val_loader is None: return None, None
    model.eval()
    total_loss, batches = 0.0, 0
    for batch in val_loader:
        input_ids = batch.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1)
            )
        total_loss += float(loss.item())
        batches += 1
    model.train()
    if batches == 0: return None, None
    avg_loss = total_loss / batches
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    return avg_loss, ppl

# --------------------------- Utils ---------------------------

def _safe_num(x):
    try: x = float(x)
    except Exception: return None
    return x if math.isfinite(x) else None

def _vec_head(t, k=16):
    v = t.detach().reshape(-1).float().cpu().tolist()[:k]
    return [round(float(x), 6) if math.isfinite(float(x)) else None for x in v]

def _global_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is None: continue
        g = p.grad.detach().float()
        total += g.pow(2).sum().item()
    return math.sqrt(total)

def pretty_name(name: str) -> str:
    if "token_embedding" in name: return "Token Embedding Matrix"
    if "pos_embedding" in name: return "Positional Embedding"
    if "attn.qkv" in name: return "QKV Projection (Query, Key, Value)"
    if "attn.c_proj" in name: return "Attention Output Projection (W₀)"
    if "ln1" in name: return "Pre-Attention LayerNorm"
    if "ln2" in name: return "Pre-FFN LayerNorm"
    if "final_norm" in name: return "Final LayerNorm"
    if "lm_head" in name: return "Output Projection"
    if name.endswith(".bias"): return "Bias Term"
    if name.endswith(".weight"): return "Weight Matrix"
    return name

# --------------------------- Logging ---------------------------

def log_optim_step(model, optimizer, scaler, global_step, lr,
                   pre_clip_gn, post_clip_gn, snaps_before=None, snaps_after=None,
                   out_path="log/current/optim.json", layer_cap=100,cached_grads=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
    b1, b2 = pg0.get("betas", (None, None))
    eps = pg0.get("eps", None)
    wd  = pg0.get("weight_decay", None)
    try: amp_enabled = bool(scaler.is_enabled())
    except Exception: amp_enabled = False
    try: amp_scale = float(scaler.get_scale())
    except Exception: amp_scale = 1.0

    parameters = []
    for name, p in model.named_parameters():
        g = None
        if cached_grads and name in cached_grads:
            g = cached_grads[name]
        elif p.grad is not None:
            g = p.grad

        st = optimizer.state.get(p, {})
        m = st.get("exp_avg", None)
        v = st.get("exp_avg_sq", None)
        entry = {
            "name": name,
            "pretty_name": pretty_name(name),
            "shape": list(p.shape),
            "num_params": p.numel(),
            "param_norm": _safe_num(p.detach().float().norm().item()),
            "grad_norm": _safe_num(g.detach().float().norm().item()) if g is not None else None,
            "m_norm":     _safe_num(m.detach().float().norm().item()) if m is not None else None,
            "v_norm":     _safe_num(v.detach().float().norm().item()) if v is not None else None,
        }
        parameters.append(entry)
        if len(parameters) >= layer_cap: break

    samples = []
    if snaps_before and snaps_after:
        for key, b in snaps_before.items():
            a = snaps_after.get(key)
            if a is None: continue
            delta = a - b
            samples.append({
                "name": key,
                "before_head": _vec_head(b, 16),
                "after_head":  _vec_head(a, 16),
                "delta_head":  _vec_head(delta, 16),
                "update_norm": _safe_num(delta.detach().float().norm().item()),
            })

    payload = {
        "step": int(global_step),
        "lr": _safe_num(lr),
        "amp": {"enabled": amp_enabled, "scaler_scale": _safe_num(amp_scale)},
        "grad": {"global_norm_pre_clip": _safe_num(pre_clip_gn),
                 "global_norm_post_clip": _safe_num(post_clip_gn)},
        "adamw": {"beta1": _safe_num(b1), "beta2": _safe_num(b2),
                  "eps": _safe_num(eps), "weight_decay": _safe_num(wd)},
        "parameters": parameters,
        "samples": samples,
        "formula": "m=β1 m+(1-β1)g; v=β2 v+(1-β2)g²; θ←θ−lr·(m̂/(√v̂+ε)+wd·θ)"
    }

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, out_path)
def _make_tokenization_payload(tokenizer, input_ids,sample_index):
    # grab one sample from the batch
    ids = input_ids[sample_index].detach().to("cpu").tolist()
    # tokens (strings)
    try:
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        tokens = None
    # best-effort text (decoded from ids)
    try:
        input_text = tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        input_text = None
    # padding mask (1 = real token, 0 = pad) — if no pad id, treat all as real
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        attention_mask = [1] * len(ids)
    else:
        attention_mask = [0 if i == pad_id else 1 for i in ids]

    return {
        "sample_index": sample_index,            # <-- include which row in the batch
        "sample": {
            "input_text": input_text,
            "tokens": tokens,
            "ids": ids,
            "attention_mask": attention_mask
        },
        "formula": "x → tokenizer(x) → tokens → ids → pad/truncate→T → attention_mask"
    } 
def _write_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")   
def log_tokenization_live(tokenizer, input_ids,out_dir: str = "log/current"):
     payload = []
     for i in range(input_ids.size(0)):
        payload.append(_make_tokenization_payload(tokenizer,input_ids,i))
        _write_jsonl(os.path.join(out_dir, "tokenization.jsonl"), payload) 
def write_loss_json(global_steps, input_ids, logits, loss_scalar, micro_loss,
                    tokenizer=None, out_path="log/current/loss.json", ignore_index=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with torch.no_grad():
        B, T, V = logits.shape
        if T < 2: return
        targets = input_ids[:, 1:].contiguous()
        pred    = logits[:, :-1, :].float()
        log_probs = pred.log_softmax(dim=-1)
        nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        mask = (targets != ignore_index).float() if ignore_index is not None else torch.ones_like(targets, dtype=torch.float32)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        per_seq = (nll * mask).sum(dim=1) / valid_counts
        nll_flat = (nll * mask).view(-1)

        loss_f = float(loss_scalar.detach().cpu().item())
        micro_f = float(micro_loss.detach().cpu().item())
        ppl = float(math.exp(loss_f))

        hardest = []
        if not (float(nll_flat.sum()) == 0.0 and int(mask.sum()) == 0):
            vals, idxs = torch.topk(nll_flat, k=min(10, nll.numel()), largest=True)
            pred_ids = pred.argmax(dim=-1)
            for v, flat_idx in zip(vals.tolist(), idxs.tolist()):
                b = flat_idx // (T-1)
                t = flat_idx %  (T-1)
                tgt_id = int(targets[b, t].item())
                pred_id = int(pred_ids[b, t].item())
                tgt_tok = tokenizer.decode([tgt_id]) if tokenizer else None
                pred_tok = tokenizer.decode([pred_id]) if tokenizer else None
                hardest.append({"b": b, "t": t+1,
                                "target_id": tgt_id, "target_token": tgt_tok,
                                "pred_id": pred_id, "pred_token": pred_tok,
                                "nll": float(v)})

        payload = {"step": int(global_steps),
                   "micro_loss": micro_f,
                   "true_loss": loss_f,
                   "ppl": ppl,
                   "batch_size": int(B),
                   "seq_len": int(T),
                   "vocab_size": int(V),
                   "per_seq_loss": [float(x) for x in per_seq.detach().cpu().tolist()],
                   "hardest_tokens": hardest}

        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, out_path)

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Step-resumable LLM trainer")
    parser.add_argument("--log_everything_path", type=str, required=True)
    parser.add_argument("--steps_today", type=int, default=None)
    args = parser.parse_args()

    cfg_path = os.path.join("configs", "custom_llm.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device","cuda")=="cuda" else "cpu")
    batch_size = int(config.get("batch_size", 2))
    grad_accum = int(config.get("grad_accum", 1))
    clip_grad = float(config.get("clip_grad", 1.0))
    weight_decay = float(config.get("weight_decay", 0.1))
    epochs = int(config.get("epochs", 1))
    log_every = int(config.get("log_every", 100))
    save_every = int(config.get("save_every", 1000))
    eval_every = int(config.get("eval_every", 0))
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    amp_enabled = bool(config.get("amp", True)) and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled)

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    out_dir = config.get("out_dir", checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)
    resume_from = config.get("resume_from", "") or os.path.join(out_dir, "ckpt_last.pt")

    data_path = config["data_path"]
    key = config.get("jsonl_key", "input_ids")
    def build_ds(path): return BinDataset(path) if path.endswith(".bin") else TextDataset(path, key=key)
    train_ds = build_ds(data_path)
    N = len(train_ds)

    val_loader = None
    if config.get("val_path"):
        val_ds = build_ds(config["val_path"])
        val_loader = DataLoader(val_ds, batch_size=int(config.get("eval_batch_size", batch_size)),
                                shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available())

    batch_sampler = ResumableEpochBatchSampler(N, batch_size, seed=seed, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    steps_per_epoch = len(batch_sampler)
    if steps_per_epoch == 0: raise ValueError("steps_per_epoch=0 — dataset too small or batch too big.")

    model_sig = signature(LLMModel.__init__)
    allowed = set(model_sig.parameters.keys()) - {"self"}
    model_cfg = {k: v for k,v in config.items() if k in allowed}
    model_cfg["log_everything_path"] = args.log_everything_path
    model = LLMModel(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-5)), weight_decay=weight_decay)

    explicit_budget = int(config.get("total_step_budget", 0))
    total_step_budget = explicit_budget if explicit_budget > 0 else (epochs * steps_per_epoch)
    lr_at = make_lr_fn(config, total_step_budget)

    global_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"🔄 Resuming from {resume_from}")
        global_step, sampler_state = load_ckpt(resume_from, model, optimizer, scaler)
        if sampler_state: batch_sampler.load_state_dict(sampler_state)
    for pg in optimizer.param_groups: pg["lr"] = lr_at(global_step)

    steps_today = args.steps_today if args.steps_today is not None else int(config.get("steps_today", 0))
    target_global_step = min(global_step+steps_today, total_step_budget) if steps_today>0 else None

    print(f"📏 Dataset N={N} | steps/epoch={steps_per_epoch} | epochs={epochs} | total={total_step_budget}")
    print(f"▶️  Starting at step={global_step} | LR={optimizer.param_groups[0]['lr']:.3e}")

    start_epoch = batch_sampler.epoch
    micro_since_step, running_loss, stop_all = 0, 0.0, False

    for epoch in range(start_epoch, epochs if explicit_budget==0 else 10**12):
        print(f"\n🚀 Epoch {epoch+1} (sampler epoch={batch_sampler.epoch})")
        stepped_this_iter = False
        for step_in_epoch, batch in enumerate(train_loader):
            for pg in optimizer.param_groups: pg["lr"] = lr_at(global_step)
            input_ids = batch.to(device, non_blocking=True)
            with autocast(enabled=amp_enabled):
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1)
                )
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            micro_since_step += 1
            running_loss += float(loss.item())
            if micro_since_step == grad_accum:
                track_keys = ["lm_head.weight","token_embedding.token_embed.weight"]
                named = dict(model.named_parameters())
                def slice16(t): return t.detach().cpu().view(-1)[:16].clone()
                snaps_before = {k: slice16(named[k]) for k in track_keys if k in named}
                scaler.unscale_(optimizer)
                pre_clip_gn = _global_grad_norm(model)
                if clip_grad>0: clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                post_clip_gn = _global_grad_norm(model)
                scaler.step(optimizer); scaler.update()
                named = dict(model.named_parameters())
                snaps_after = {k: slice16(named[k]) for k in snaps_before}
                cached_grads = {name: (p.grad.detach().clone() if p.grad is not None else None) 
                for name, p in model.named_parameters()}
                optimizer.zero_grad(set_to_none=True)
                global_step += 1; micro_since_step = 0; stepped_this_iter = True
                last_input_ids, last_logits, last_loss = input_ids, logits, loss
                last_snaps_before, last_snaps_after = snaps_before, snaps_after
                last_pre_clip_gn, last_post_clip_gn = pre_clip_gn, post_clip_gn
                if log_every and (global_step%log_every==0):
                    disp_loss = running_loss*grad_accum; running_loss=0.0
                    print(f"step {global_step} | lr {optimizer.param_groups[0]['lr']:.3e} | loss {disp_loss:.4f}")
                if eval_every and val_loader is not None and (global_step%eval_every==0):
                    vloss,ppl = evaluate_model(model,val_loader,device,amp_enabled)
                    if vloss is not None: print(f"[eval] step {global_step} | val_loss {vloss:.4f} | ppl {ppl:.2f}")
                if save_every and (global_step%save_every==0):
                    ckpt_path=os.path.join(out_dir,f"ckpt_{global_step}.pt")
                    save_ckpt(ckpt_path,model,optimizer,scaler,global_step,batch_sampler)
                    save_ckpt(os.path.join(out_dir,"ckpt_last.pt"),model,optimizer,scaler,global_step,batch_sampler)
                    print(f"💾 Saved {ckpt_path} and updated ckpt_last.pt")
                if target_global_step is not None and global_step>=target_global_step:
                    print(f"⏹ Reached steps_today target ({steps_today})."); stop_all=True
                if global_step>=total_step_budget:
                    print(f"⏹ Reached total_step_budget ({total_step_budget})."); stop_all=True
            if stop_all: break
        if stepped_this_iter:
            epoch_ckpt=os.path.join(out_dir,f"ckpt_epoch{epoch+1}.pt")
            save_ckpt(epoch_ckpt,model,optimizer,scaler,global_step,batch_sampler)
            save_ckpt(os.path.join(out_dir,"ckpt_last.pt"),model,optimizer,scaler,global_step,batch_sampler)
            print(f"✅ Epoch {epoch+1} done | saved {epoch_ckpt} and updated ckpt_last.pt")
        if stop_all: break
        batch_sampler.set_epoch(epoch+1)

    # Final logging + save
    loss_scalar = last_loss * grad_accum
    if config.get("log_everything", True):
        tokenizer_config={"model_type":config.get("model_type","generic"),
                          "tokenizer_path":config.get("tokenizer_path"),
                          "tokenizer_type":config.get("tokenizer_type"),
                          "max_position_embeddings":config.get("max_position_embeddings")}
        tokenizer = TokenizerManager(tokenizer_config).tokenizer
        log_tokenization_live(tokenizer,input_ids,out_dir=args.log_everything_path)
        log_optim_step(model,optimizer,scaler,global_step,
                       optimizer.param_groups[0]["lr"],
                       last_pre_clip_gn,last_post_clip_gn,
                       last_snaps_before,last_snaps_after,
                       out_path=os.path.join(args.log_everything_path,"optim.json"),cached_grads=cached_grads)
        write_loss_json(global_step,last_input_ids,last_logits,
                        loss_scalar,last_loss,
                        tokenizer=tokenizer,
                        out_path=os.path.join(args.log_everything_path,"loss.json"))

    final_ckpt=os.path.join(out_dir,"ckpt_last.pt")
    save_ckpt(final_ckpt,model,optimizer,scaler,global_step,batch_sampler)
    print(f"🏁 Training complete. step={global_step}, ckpt={final_ckpt}")

if __name__=="__main__":
    main()
