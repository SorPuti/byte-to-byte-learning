# Anton — LLM from Scratch (Python + NumPy)

**Date:** 2026-11-10  
**Inspired by:** "I Trained an LLM from Scratch" (YouTube, ID: vKdKtxpfvIc)  
**Language:** Python 3.14 + NumPy 2.4 (no ML framework)  
**Goal:** Functional GPT-style language model (~6M params) trained on Shakespeare, generates Shakespeare-style text.

---

## 1. Project Structure

```
byte-to-byte-learning/
├── data/
│   ├── shakespeare.txt        # downloaded once at first run
│   └── tokenizer.json         # saved BPE vocab + merge rules
├── src/
│   ├── __init__.py
│   ├── tokenizer.py           # BPETokenizer: train / encode / decode / save / load
│   ├── model.py               # Transformer: forward + manual backward
│   ├── optimizer.py           # AdamW
│   ├── dataset.py             # batch sampler
│   └── generate.py            # autoregressive sampling
├── checkpoints/               # created at runtime
├── train.py                   # entry point: download → tokenize → train loop
├── config.py                  # all hyperparameters
└── requirements.txt           # numpy only
```

---

## 2. Hyperparameters (`config.py`)

| Name | Value | Notes |
|---|---|---|
| `vocab_size` | 10 000 | target after BPE merges |
| `num_merges` | 10 000 | BPE training iterations |
| `context_length` | 256 | tokens per training window |
| `d_model` | 256 | embedding dimension |
| `n_heads` | 4 | attention heads (head_dim = 64) |
| `n_layers` | 4 | transformer blocks |
| `d_ff` | 1 024 | feed-forward hidden dim (4× d_model) |
| `batch_size` | 64 | windows per step |
| `learning_rate` | 3e-4 | AdamW lr |
| `weight_decay` | 0.1 | AdamW (applied to weight matrices only) |
| `beta1` | 0.9 | AdamW momentum |
| `beta2` | 0.999 | AdamW velocity |
| `eps` | 1e-8 | AdamW numerical stability |
| `max_steps` | 20 000 | training steps |
| `val_interval` | 500 | steps between validation prints |
| `ckpt_interval` | 2 000 | steps between checkpoint saves |
| `val_fraction` | 0.1 | fraction of corpus held out for validation |

Estimated total parameters: ~5.8M (see Section 4 for breakdown).

---

## 3. Component Specifications

### 3.1 BPE Tokenizer (`src/tokenizer.py`)

**Training:**
1. Build initial vocabulary from unique characters in corpus (~100 chars). Each character gets a unique integer ID.
2. Encode corpus as list of character IDs.
3. Repeat `num_merges` times:
   a. Count all adjacent pair frequencies in the current token sequence.
   b. Find the most frequent pair.
   c. Add the pair as a new token (ID = current vocab size).
   d. Replace all occurrences of the pair in the sequence with the new token.
4. Store: `vocab` (id → string), `merges` (list of (a, b) pairs in order).

**Encode (text → list[int]):**
1. Split text into characters, convert to IDs.
2. Apply merge rules in training order: scan for each pair, replace with merged token.

**Decode (list[int] → str):**
1. Look up each ID in `vocab`, concatenate.

**Persistence:**
- Save/load as JSON: `{"vocab": {id: str}, "merges": [[a, b], ...]}`.

**Complexity:** Training is O(num_merges × corpus_length). For Shakespeare (~1MB) and 10k merges, expect ~1-2 min on CPU.

---

### 3.2 Transformer Model (`src/model.py`)

Each layer is a Python class with `forward(x)` → returns output and stores cache, and `backward(dout)` → returns `dx` and accumulates `dW`, `db`.

**Layer hierarchy:**

```
Transformer
├── TokenEmbedding        (vocab_size × d_model)
├── PositionalEmbedding   (context_length × d_model)
├── TransformerBlock × n_layers
│   ├── LayerNorm
│   ├── CausalSelfAttention
│   │   ├── Linear Wq   (d_model × d_model)
│   │   ├── Linear Wk   (d_model × d_model)
│   │   ├── Linear Wv   (d_model × d_model)
│   │   └── Linear Wo   (d_model × d_model)
│   ├── LayerNorm
│   └── FeedForward
│       ├── Linear W1   (d_model × d_ff)
│       ├── GELU
│       └── Linear W2   (d_ff × d_model)
└── LayerNorm final
   (output projection shares weights with TokenEmbedding — weight tying)
```

**Forward pass details:**

- `CausalSelfAttention`: computes Q, K, V; scaled dot-product attention; applies causal mask (upper triangle = -inf before softmax); multi-head split/merge via reshape.
- `LayerNorm`: normalizes over last axis; learnable scale (γ) and shift (β).
- `GELU`: approximation `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.
- Weight tying: the final logit projection reuses `TokenEmbedding.W.T` — saves ~2.5M params and improves convergence.

**Backward pass — what each layer caches:**

| Layer | Cached for backward |
|---|---|
| `Linear` | input `x`, weight `W` |
| `Softmax` | output `s` |
| `CausalSelfAttention` | Q, K, V, attn_weights, pre-mask scores |
| `LayerNorm` | input `x`, normalized `x_hat`, variance, γ |
| `GELU` | input `x` |
| `Embedding` | token indices (gradient is sparse scatter) |

**Loss:** Cross-entropy over all positions. `loss = -log_softmax(logits)[targets]`, averaged over batch and time.

**Parameter count breakdown (~5.8M total):**

| Component | Params |
|---|---|
| Token embedding (10k × 256) | 2 560 000 |
| Positional embedding (256 × 256) | 65 536 |
| Per block: 4 attention linears (4 × 256²) | 262 144 |
| Per block: FFN linears (256×1024 + 1024×256) | 524 288 |
| Per block: 2 LayerNorms (2 × 2 × 256) | 1 024 |
| 4 blocks total | 3 149 824 |
| Final LayerNorm | 512 |
| Output projection | 0 (tied) |
| **Total** | **~5 775 872** |

---

### 3.3 AdamW Optimizer (`src/optimizer.py`)

Standard AdamW. On construction, initializes `m` and `v` dicts (zero arrays shaped like each parameter). On `step()`:

1. Increment timestep `t`.
2. For each `(param, grad)`:
   - Apply weight decay to `param` (only if param is a weight matrix, not bias/LayerNorm).
   - Update `m[param] = β1 * m + (1-β1) * grad`.
   - Update `v[param] = β2 * v + (1-β2) * grad²`.
   - Bias-correct: `m_hat = m / (1-β1^t)`, `v_hat = v / (1-β2^t)`.
   - `param -= lr * m_hat / (√v_hat + ε)`.

Params are identified by their Python `id()` to key the moment dicts.

---

### 3.4 Dataset / Batch Sampler (`src/dataset.py`)

```python
def get_batch(token_ids, batch_size, context_length, rng):
    # token_ids: 1D numpy array of all training IDs
    # returns x (B, T), targets (B, T) — targets = x shifted right by 1
    starts = rng.integers(0, len(token_ids) - context_length - 1, size=batch_size)
    x       = np.stack([token_ids[i   : i+context_length  ] for i in starts])
    targets = np.stack([token_ids[i+1 : i+context_length+1] for i in starts])
    return x, targets
```

Validation uses a fixed seed for reproducibility.

---

### 3.5 Autoregressive Generation (`src/generate.py`)

```python
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0):
    ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        context = ids[-context_length:]          # trim to max context
        logits, _ = model.forward(context)       # (1, T, vocab)
        logits_last = logits[0, -1, :] / temperature
        probs = softmax(logits_last)
        next_id = np.random.choice(len(probs), p=probs)
        ids.append(next_id)
    return tokenizer.decode(ids)
```

CLI usage:
```bash
python generate.py --checkpoint checkpoints/step_20000.npz \
                   --prompt "To be or not" --tokens 200 --temp 0.8
```

---

### 3.6 Training Entry Point (`train.py`)

```
1. Load config from config.py
2. Download shakespeare.txt if not present (from Project Gutenberg)
3. If tokenizer.json exists: load. Else: train BPE, save.
4. Tokenize full corpus. Split 90% train / 10% val.
5. Initialize Transformer(config) with Xavier/He init.
6. Initialize AdamW(model.parameters(), lr, wd, β1, β2, ε)
7. Training loop (step = 0..max_steps):
   a. get_batch(train_ids, ...) → x, targets
   b. logits, loss = model.forward(x, targets)
   c. model.backward()
   d. optimizer.step(model.parameters_and_grads())
   e. model.zero_grad()
   f. If step % val_interval == 0:
      - Compute val loss (no grad needed: forward only)
      - Print: step, train_loss, val_loss, tokens/sec
      - Sample 100 tokens from prompt "ROMEO:" and print
   g. If step % ckpt_interval == 0:
      - Save model weights + optimizer state to checkpoints/step_XXXXX.npz
8. Print final sample.
```

---

## 4. Data Flow Diagram

```
shakespeare.txt (1MB)
        │
        ▼
  BPETokenizer.train()  ← 10k merges, ~1-2 min
        │
        ▼
  token_ids: np.array of shape (N,)  ← N ≈ 500k–1M tokens
        │
   ┌────┴────┐
  90%      10%
 train      val
        │
        ▼
  get_batch() → x (64,256), targets (64,256)
        │
        ▼
  Transformer.forward() → loss (scalar)
        │
        ▼
  Transformer.backward() → grads on all params
        │
        ▼
  AdamW.step() → updated params
        │
        └──── repeat 20k steps (~11h CPU)
```

---

## 5. Error Handling and Robustness

- **Checkpoint resume:** `train.py` checks for the latest `checkpoints/step_*.npz` on startup and resumes from there if found.
- **Numerical stability:** LayerNorm and softmax use `x - x.max()` shift before exp to prevent overflow.
- **Loss spike detection:** if `loss > 10.0` at any step, print a warning (NaN/inf often follows from bad init or too-high LR).
- **Tokenizer cache:** BPE training is skipped if `data/tokenizer.json` already exists.

---

## 6. Out of Scope

- GPU support (next episode's concern — PyTorch port)
- Flash attention
- Beam search / top-k / top-p sampling (temperature-only for now)
- Multi-dataset training
- Distributed training
