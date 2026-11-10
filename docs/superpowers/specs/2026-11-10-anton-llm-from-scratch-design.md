# Anton — LLM from Scratch (Python + NumPy)

**Date:** 2026-11-10  
**Inspired by:** "I Trained an LLM from Scratch" (YouTube, ID: vKdKtxpfvIc)  
**Language:** Python 3.14 + NumPy 2.4 (no ML framework)  
**Goal:** Functional GPT-style language model (~6M params) trained on Machado de Assis (literatura brasileira, domínio público), generates Portuguese prose in his style.

---

## 1. Project Structure

```
byte-to-byte-learning/
├── data/
│   ├── shakespeare.txt        # downloaded once at first run
│   └── tokenizer.json         # saved BPE vocab + merge rules
├── src/
│   ├── __init__.py
│   ├── tokenizer.py           # Tokenizer: train / encode / decode / save / load
│   ├── gpt.py                 # Transformer: forward + manual backward
│   ├── optimizer.py           # AdamW com LR schedule
│   ├── dataset.py             # batch sampler por época
│   └── generate.py            # autoregressive sampling
├── checkpoints/               # created at runtime
├── train.py                   # entry point: download → tokenize → train loop
├── config.py                  # all hyperparameters
└── requirements.txt           # numpy only
```

---

## 2. Hyperparameters (`config.py`)

| Name | Valor | Notas |
|---|---|---|
| `VOCAB_SIZE` | 10 000 | alvo após merges BPE |
| `num_merges` | 10 000 | iterações de treino BPE |
| `SEQUENCE_LENGTH` | 128 | tokens por janela de treino |
| `EMBEDDING_DIM` | 200 | dimensão dos embeddings |
| `N_HEADS` | 5 | cabeças de atenção (head_dim = 40) |
| `N_BLOCKS` | 5 | blocos transformer |
| `d_ff` | 800 | hidden dim FFN (4× EMBEDDING_DIM) |
| `BATCH_SIZE` | 64 | janelas por batch |
| `LEARNING_RATE_MAX` | 0.001 | lr inicial (cosine annealing) |
| `LEARNING_RATE_MIN` | 0.0001 | lr mínimo (cosine annealing) |
| `WEIGHT_DECAY` | 0.1 | AdamW (apenas matrizes de peso) |
| `BETA1` | 0.9 | AdamW momentum |
| `BETA2` | 0.95 | AdamW velocity |
| `eps` | 1e-8 | estabilidade numérica AdamW |
| `EPOCHS` | 75 | épocas de treino |
| `VALIDATION_BATCHES` | 10 | batches usados para validação por época |

> **Nota:** O treino usa épocas (`EPOCHS=75`) com cosine annealing de LR entre `LEARNING_RATE_MAX` e `LEARNING_RATE_MIN`. Usa `tqdm` para barra de progresso.

Estimated total parameters: ~4.4M (see Section 3.2 for breakdown).

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

**Complexity:** Training is O(num_merges × corpus_length). Para o corpus de Machado de Assis (~1MB) com 10k merges, esperar ~1-2 min na CPU.

---

### 3.2 Transformer Model (`src/gpt.py`)

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

**Parameter count breakdown (~4.4M total):**

| Component | Params |
|---|---|
| Token embedding (10k × 200) | 2 000 000 |
| Positional embedding (128 × 200) | 25 600 |
| Per block: 4 attention linears (4 × 200²) | 160 000 |
| Per block: FFN linears (200×800 + 800×200) | 320 000 |
| Per block: 2 LayerNorms (2 × 2 × 200) | 800 |
| 5 blocks total | 2 404 000 |
| Final LayerNorm | 400 |
| Output projection | 0 (tied) |
| **Total** | **~4 430 000** |

---

### 3.3 AdamW Optimizer (`src/optimizer.py`)

AdamW com cosine annealing de learning rate. Inicializa `m` e `v` (zero arrays com shape de cada parâmetro). No `step(lr)`:

1. Incrementa timestep `t`.
2. Para cada `(param, grad)`:
   - Aplica weight decay em `param` (apenas matrizes de peso, não biases/LayerNorm).
   - `m = β1*m + (1-β1)*grad`
   - `v = β2*v + (1-β2)*grad²`
   - Bias-correct: `m_hat = m / (1-β1^t)`, `v_hat = v / (1-β2^t)`
   - `param -= lr * m_hat / (√v_hat + ε)`

**Cosine annealing** (computado no `train.py`):
```python
lr = LR_MIN + 0.5*(LR_MAX - LR_MIN)*(1 + cos(π * step / total_steps))
```

Params identificados pelo `id()` Python para indexar os dicts de momento.

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
python generate.py --checkpoint checkpoints/epoch_75.npz \
                   --prompt "— Mas, senhora" --tokens 200 --temp 0.8
```

---

### 3.6 Training Entry Point (`train.py`)

```
1. Carrega config de config.py
2. Baixa corpus Machado de Assis se não existir (Project Gutenberg PT)
3. Se tokenizer.json existir: carrega. Senão: treina BPE, salva.
4. Tokeniza corpus completo. Separa 90% treino / 10% val.
5. Inicializa GPT(config) com init Xavier/He.
6. Inicializa AdamW(model.parameters(), wd, β1, β2, ε)
7. Loop de treino (epoch = 0..EPOCHS, com tqdm):
   a. Calcula lr via cosine annealing para o passo atual
   b. get_batch(train_ids, ...) → x, targets
   c. logits, loss = model.forward(x, targets)
   d. model.backward()
   e. optimizer.step(lr, model.parameters_and_grads())
   f. model.zero_grad()
   h. Ao fim de cada época:
      - Calcula val loss em VALIDATION_BATCHES batches (só forward)
      - Imprime: epoch, train_loss, val_loss, tokens/sec
      - Amostra 100 tokens do prompt "— Mas, senhora" e imprime
      - Salva checkpoint em checkpoints/epoch_XX.npz
8. Imprime amostra final.
```

---

## 4. Data Flow Diagram

```
machado_assis.txt (1MB)
        │
        ▼
  Tokenizer.train()  ← 10k merges, ~1-2 min
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
        └──── repeat 75 epochs
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
