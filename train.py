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

# --------------------------- Optional dataset helpers ---------------------------

# Minimal BinDataset (uint16, shape: (N, seq_len)) — only used if you pass a .bin
class BinDataset(Dataset):
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

# If you already have your own TextDataset, import it; else fall back to a tiny inline one.
try:
    from src.data.text_dataset import TextDataset  # expects pretokenized=True jsonl with a single int-id list per line
except Exception:
    class TextDataset(Dataset):
        """
        Minimal JSONL loader: each line is {"input_ids": [int, ...]} and length == seq_len.
        """
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

from src.model.llm_model import LLMModel    # your model
try:
    from src.tokenizer.tokenizer_manager import TokenizerManager
except Exception:
    TokenizerManager = None  # not required for pretokenized data

# --------------------------- Resumable per-epoch sampler ---------------------------

class ResumableEpochBatchSampler(torch.utils.data.Sampler):
    """
    Deterministic shuffle each epoch (seed + epoch), exact resume via (epoch, batch_pos).
    Yields the remaining batches of the CURRENT epoch; call set_epoch(epoch+1) to reshuffle.
    """
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

# --------------------------- Evaluation ---------------------------

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

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Step-resumable LLM trainer")
    parser.add_argument("--model", type=str, required=True, help="Model name for configs/<model>.json")
    parser.add_argument("--steps_today", type=int, default=None, help="Do at most N optimizer steps this run (0/None = no cap)")
    args = parser.parse_args()

    # Load config
    cfg_path = os.path.join("configs", f"{args.model}.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Basic knobs
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "cuda") == "cuda" else "cpu")
    batch_size = int(config.get("batch_size", 2))
    grad_accum = int(config.get("grad_accum", 1))
    clip_grad = float(config.get("clip_grad", 1.0))
    weight_decay = float(config.get("weight_decay", 0.1))
    epochs = int(config.get("epochs", 1))
    log_every = int(config.get("log_every", 100))
    save_every = int(config.get("save_every", 1000))
    eval_every = int(config.get("eval_every", 0))  # 0 = off
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # AMP
    amp_enabled = bool(config.get("amp", True)) and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled)

    # I/O
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    out_dir = config.get("out_dir", checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)
    resume_from = config.get("resume_from", "") or os.path.join(out_dir, "ckpt_last.pt")

    # ----------------- Build datasets -----------------
    data_path = config["data_path"]
    seq_len = int(config.get("max_position_embeddings", 2048))
    key = config.get("jsonl_key", "input_ids")
    val_path = config.get("val_path", None)

    def build_ds(path):
        if path.endswith(".bin"):
            return BinDataset(path)
        else:
            # expect pretokenized jsonl with fixed-length ids
            return TextDataset(path, key=key)

    train_ds = build_ds(data_path)
    N = len(train_ds)

    # Optional validation
    val_loader = None
    if val_path:
        val_ds = build_ds(val_path)
        val_loader = DataLoader(val_ds, batch_size=int(config.get("eval_batch_size", batch_size)),
                                shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available())

    # ----------------- Sampler / DataLoader -----------------
    batch_sampler = ResumableEpochBatchSampler(
        dataset_size=N, batch_size=batch_size, seed=seed, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=batch_sampler, num_workers=0, pin_memory=torch.cuda.is_available()
    )

    steps_per_epoch = len(batch_sampler)  # with drop_last=True: floor(N / B)
    if steps_per_epoch == 0:
        raise ValueError("steps_per_epoch computed as 0 — increase dataset size or lower batch_size.")

    # ----------------- Model / Optimizer -----------------
    model = LLMModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-5)), weight_decay=weight_decay)

    # ----------------- Step budgets & LR schedule -----------------
    explicit_budget = int(config.get("total_step_budget", 0))
    total_step_budget = explicit_budget if explicit_budget > 0 else (epochs * steps_per_epoch)
    lr_at = make_lr_fn(config, total_step_budget)

    # ----------------- Resume -----------------
    global_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"🔄 Resuming from {resume_from}")
        global_step, sampler_state = load_ckpt(resume_from, model, optimizer, scaler)
        if sampler_state: batch_sampler.load_state_dict(sampler_state)

    # set LR to match restored global_step
    for pg in optimizer.param_groups:
        pg["lr"] = lr_at(global_step)

    # Daily step cap
    steps_today = args.steps_today if args.steps_today is not None else int(config.get("steps_today", 0))
    target_global_step = None
    if steps_today and steps_today > 0:
        target_global_step = min(global_step + steps_today, total_step_budget)

    # ----------------- Training loop -----------------
    print(f"📏 Dataset N={N} | steps/epoch={steps_per_epoch} | epochs={epochs} | total_step_budget={total_step_budget}")
    print(f"▶️  Starting at global_step={global_step} | LR={optimizer.param_groups[0]['lr']:.3e}")

    start_epoch = batch_sampler.epoch
    micro_since_step = 0
    running_loss = 0.0

    stop_all = False
    for epoch in range(start_epoch, epochs if explicit_budget == 0 else 10**12):  # large upper bound if budget drives stop
        print(f"\n🚀 Epoch {epoch + 1} (sampler epoch={batch_sampler.epoch})")
        stepped_this_iter = False

        for step_in_epoch, batch in enumerate(train_loader):
            # Keep LR in sync *for this optimizer step* (based on current global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(global_step)

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

            stepped_this_iter = False
            if micro_since_step == grad_accum:
                scaler.unscale_(optimizer)
                if clip_grad and clip_grad > 0:
                    clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                micro_since_step = 0
                stepped_this_iter = True

                # logging
                if log_every and (global_step % log_every == 0):
                    disp_loss = running_loss * grad_accum  # undo the scale for display
                    running_loss = 0.0
                    print(f"step {global_step} | lr {optimizer.param_groups[0]['lr']:.3e} | loss {disp_loss:.4f}")

                # eval
                if eval_every and val_loader is not None and (global_step % eval_every == 0):
                    vloss, ppl = evaluate_model(model, val_loader, device, amp_enabled)
                    if vloss is not None:
                        print(f"[eval] step {global_step} | val_loss {vloss:.4f} | ppl {ppl:.2f}")

                # periodic save (only after a full optimizer step)
                if save_every and (global_step % save_every == 0):
                    ckpt_path = os.path.join(out_dir, f"ckpt_{global_step}.pt")
                    save_ckpt(ckpt_path, model, optimizer, scaler, global_step, batch_sampler=batch_sampler)
                    ckpt_last = os.path.join(out_dir, "ckpt_last.pt")
                    save_ckpt(ckpt_last, model, optimizer, scaler, global_step, batch_sampler=batch_sampler)
                    print(f"💾 Saved {ckpt_path} and updated ckpt_last.pt")

                # stop by daily cap or total budget
                if target_global_step is not None and global_step >= target_global_step:
                    print(f"⏹ Reached steps_today target ({steps_today}).")
                    stop_all = True
                if global_step >= total_step_budget:
                    print(f"⏹ Reached total_step_budget ({total_step_budget}).")
                    stop_all = True

            if stop_all:
                break

        # end-of-epoch: optional eval + save, then reshuffle for next epoch
        if eval_every == 0 and val_loader is not None:
            vloss, ppl = evaluate_model(model, val_loader, device, amp_enabled)
            if vloss is not None:
                print(f"[eval:epoch] {epoch+1} | val_loss {vloss:.4f} | ppl {ppl:.2f}")

        # Save an epoch checkpoint at a safe boundary (only if we actually stepped)
        if stepped_this_iter:
            epoch_ckpt = os.path.join(out_dir, f"ckpt_epoch{epoch+1}.pt")
            save_ckpt(epoch_ckpt, model, optimizer, scaler, global_step, batch_sampler=batch_sampler)
            ckpt_last = os.path.join(out_dir, "ckpt_last.pt")
            save_ckpt(ckpt_last, model, optimizer, scaler, global_step, batch_sampler=batch_sampler)
            print(f"✅ Epoch {epoch+1} done | saved {epoch_ckpt} and updated ckpt_last.pt")

        if stop_all:
            break

        # new epoch shuffle
        batch_sampler.set_epoch(epoch + 1)

    # final save
    final_ckpt = os.path.join(out_dir, "ckpt_last.pt")
    save_ckpt(final_ckpt, model, optimizer, scaler, global_step, batch_sampler=batch_sampler)
    print(f"🏁 Training complete or paused. global_step={global_step}. Final checkpoint: {final_ckpt}")

if __name__ == "__main__":
    main()
