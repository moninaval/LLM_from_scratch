# LLM Training Pipeline

This repository contains scripts and configurations for **training a Large Language Model (LLM)** with a resumable, step-based approach. It includes tools to prepare datasets, tokenize text, and run training in a flexible, GPU-availability-friendly manner.

---

## Overview of Workflow

1. **Prepare your raw dataset**  
   - This can be plain text, instruction-style JSONL (like Alpaca), or chat-formatted templates.
   - If your dataset is JSONL (e.g., instruction format), you may need to render it into plain text before tokenization.

2. **Tokenize your dataset**  
   - Uses your chosen tokenizer (HuggingFace or custom BPE) to convert raw text into token IDs.
   - Stores results in `.jsonl` or `.bin` format for efficient training.

3. **Train the model**  
   - Loads the tokenized dataset and trains the model.
   - Supports **step-based continuation** so you can train for a fixed number of steps per session (e.g., 5 hours GPU time) and resume later.

---

## Configuration

All settings are in `configs/{model_name}.json`.  
Key fields:

| Field | Description |
|-------|-------------|
| `model_name` | Identifier for your model (e.g., `phi-2`). |
| `tokenizer_path` | Path to existing tokenizer or pretrained model's tokenizer. |
| `tokenizer_type` | `huggingface` or `bpe` for custom tokenizers. |
| `data_path` | Path to **raw dataset** (before tokenization). |
| `save_bin` | `true` to store tokenized data in `.bin` format for speed. |
| `batch_size` | Number of sequences per batch. |
| `lr` | Base learning rate. |
| `epochs` | Used if `total_step_budget` is 0. |
| `total_step_budget` | Total training steps to run (overrides `epochs`). |
| `grad_accum` | Gradient accumulation steps. |
| `save_every` | Save checkpoint every N steps. |
| `eval_every` | Evaluate every N steps. |
| `checkpoint_dir` | Directory for checkpoints. |
| `out_dir` | Directory for logs/outputs. |

---

## Script 1: Tokenize Dataset

**File:** `scripts/tokenize_dataset.py`  
**Purpose:** Takes your raw dataset and converts it into `.jsonl` or `.bin` format for training.

**Run:**
```bash
# Example for HuggingFace tokenizer
python scripts/tokenize_dataset.py \
  --model my_model \
  --val_ratio 0.01 \
  --shuffle \
  --seed 42 \
  --emit_jsonl
# Example for custom tokenizer
python scripts/tokenize_dataset.py --model my-llm
```

**Notes:**
- Input path is set in `configs/{model}.json` → `data_path`.
- Tokenizer is loaded from `tokenizer_path`.
- Output file will be auto-named like:  
  `data/{basename}.cached.{seq_len}.{timestamp}.jsonl`
- If `save_bin` is `true`, a `.bin` file is also created.

---

## Script 2: Train Model

**File:** `train.py`  
**Purpose:** Trains the model on the tokenized dataset.

**Run:**
```bash
# Train using epoch-based loop (total_step_budget=0)
python train.py --model phi-2

# Train using step-based budget (for limited daily GPU time)
python train.py --model phi-2 --steps_today 1000
```

**Step-based Training Workflow:**
1. Set `total_step_budget` in config to desired global steps.
2. Each day, run training with `--steps_today N` based on available GPU hours.
3. The trainer resumes from the last checkpoint automatically.

---

## Typical End-to-End Example

```bash
# 1. Tokenize raw dataset
python scripts/tokenize_dataset.py --model phi-2

# 2. Train model for 1500 steps today
python train.py --model phi-2 --steps_today 1500

# 3. Next day, resume for another 2000 steps
python train.py --model phi-2 --steps_today 2000
```

---

## Best Practices

- **Shuffle dataset** during training for better generalization.
- **Save checkpoints** regularly (`save_every`) to prevent loss on GPU failures.
- **Use cosine LR schedule** for smoother convergence.
- Keep a **separate validation dataset** if possible.

---
