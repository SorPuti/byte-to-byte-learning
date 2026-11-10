# Anton LLM — Portuguese Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build um GPT de ~4.4M parâmetros treinado em prosa de Machado de Assis que gera texto em português, usando apenas Python + NumPy sem frameworks de ML.

**Architecture:** BPE tokenizer manual → transformer GPT com backprop escrito na mão → AdamW com cosine annealing; treino por épocas (EPOCHS=75) com checkpoints por época.

**Tech Stack:** Python 3.14, NumPy 2.4, tqdm, pytest

**Spec:** `docs/superpowers/specs/2026-11-10-anton-llm-from-scratch-design.md`

## Global Constraints

- Zero imports de torch/tensorflow/jax/keras em qualquer arquivo `src/` ou `train.py`
- `requirements.txt` lista apenas: numpy, tqdm, pytest
- Corpus: português de Machado de Assis (Project Gutenberg)
- Nomes de variáveis e constantes seguem o vídeo: `VOCAB_SIZE=10_000`, `EMBEDDING_DIM=200`, `N_HEADS=5`, `N_BLOCKS=5`, `SEQUENCE_LENGTH=128`, `BATCH_SIZE=64`, `EPOCHS=75`, `VALIDATION_BATCHES=10`, `LEARNING_RATE_MAX=0.001`, `LEARNING_RATE_MIN=0.0001`, `BETA1=0.9`, `BETA2=0.95`, `WEIGHT_DECAY=0.1`
- Modelo recebe `vocab_size = tokenizer.vocab_size` (tamanho real após BPE), não o constante `VOCAB_SIZE`
- Checkpoints salvos em `checkpoints/epoch_XX.npz`
- Prompt padrão de amostragem: `"— Mas, senhora"`

## Review Focus

- **Gradientes errados em backprop manual:** bug silencioso mais provável — cada camada com backward precisa de finite-difference check em Task 4 e Task 5.
- **Vazamento de atenção futura:** máscara causal errada faz o modelo trapacear — test em Task 5 verifica que `attn[i,j]=0` para `j>i`.
- **BPE encode/decode não-idempotente:** decode(encode(text)) deve ser idêntico ao original — test em Task 2.
- **NaN/inf em softmax e LayerNorm:** inputs com valores grandes explodem — test em Task 4 passa vetor com max=100.
- **Weight tying no backward:** embedding e head de saída compartilham `W`; gradiente deve acumular das duas direções — test em Task 6.

---

## File Structure

```
byte-to-byte-learning/
├── config.py                    # Task 1 — todas as constantes
├── requirements.txt             # Task 1
├── train.py                     # Task 8 — loop principal
├── src/
│   ├── __init__.py              # Task 1 (vazio)
│   ├── tokenizer.py             # Task 2 — classe Tokenizer (BPE)
│   ├── dataset.py               # Task 3 — get_batch()
│   ├── gpt.py                   # Tasks 4-6 — camadas + GPT completo
│   ├── optimizer.py             # Task 7 — AdamW + cosine_lr
│   └── generate.py              # Task 9 — amostragem + CLI
├── data/                        # criado em runtime
├── checkpoints/                 # criado em runtime
└── tests/
    ├── test_tokenizer.py        # Task 2
    ├── test_dataset.py          # Task 3
    ├── test_layers.py           # Tasks 4-5
    ├── test_gpt.py              # Task 6
    ├── test_optimizer.py        # Task 7
    └── test_generate.py         # Task 9
```

---

## Task 1: Scaffold

**Files:**
- Create: `config.py`
- Create: `requirements.txt`
- Create: `src/__init__.py`

**Interfaces:**
- Produces: `from config import *` expõe todas as constantes listadas em Global Constraints

- [ ] **Step 1: Criar `requirements.txt`**

  Conteúdo exato:
  ```
  numpy
  tqdm
  pytest
  ```

- [ ] **Step 2: Criar `src/__init__.py`** vazio.

- [ ] **Step 3: Criar `config.py`** com todas as constantes de Global Constraints, mais:
  ```python
  DATA_DIR   = "data"
  CORPUS_FILE = "data/machado_assis.txt"
  TOKENIZER_FILE = "data/tokenizer.json"
  CHECKPOINT_DIR = "checkpoints"
  SAMPLE_PROMPT  = "— Mas, senhora"
  SAMPLE_TOKENS  = 100
  ```

- [ ] **Step 4: Instalar dependências**

  ```bash
  pip install -r requirements.txt
  ```

- [ ] **Step 5: Verificar importação**

  ```bash
  python -c "from config import EMBEDDING_DIM; print(EMBEDDING_DIM)"
  ```
  Expected: `200`

- [ ] **Step 6: Commit**

  ```bash
  git add config.py requirements.txt src/__init__.py
  git commit -m "feat: scaffold — config and requirements"
  ```

---

## Task 2: BPE Tokenizer

**Files:**
- Create: `src/tokenizer.py`
- Create: `tests/test_tokenizer.py`

**Interfaces:**
- Produces:
  ```python
  class Tokenizer:
      vocab_size: int
      def train(self, text: str, num_merges: int) -> None
      def encode(self, text: str) -> list[int]
      def decode(self, ids: list[int]) -> str
      def save(self, path: str) -> None
      @classmethod
      def load(cls, path: str) -> "Tokenizer"
  ```

- [ ] **Step 1: Escrever os testes**

  ```python
  # tests/test_tokenizer.py
  import pytest
  from src.tokenizer import Tokenizer

  SAMPLE = "Dom Casmurro nasceu numa tarde de verão. Dom Casmurro foi embora."

  def test_vocab_grows_with_merges():
      tok = Tokenizer()
      tok.train(SAMPLE, num_merges=10)
      assert tok.vocab_size > len(set(SAMPLE))

  def test_encode_decode_roundtrip():
      tok = Tokenizer()
      tok.train(SAMPLE, num_merges=10)
      assert tok.decode(tok.encode(SAMPLE)) == SAMPLE

  def test_encode_returns_ints():
      tok = Tokenizer()
      tok.train(SAMPLE, num_merges=5)
      ids = tok.encode("Dom")
      assert all(isinstance(i, int) for i in ids)

  def test_save_load_roundtrip(tmp_path):
      tok = Tokenizer()
      tok.train(SAMPLE, num_merges=10)
      path = str(tmp_path / "tok.json")
      tok.save(path)
      tok2 = Tokenizer.load(path)
      assert tok2.decode(tok2.encode(SAMPLE)) == SAMPLE
      assert tok2.vocab_size == tok.vocab_size

  def test_stability_large_values():
      # encode/decode em texto com todos os chars ASCII imprimíveis
      text = "".join(chr(i) for i in range(32, 127)) * 3
      tok = Tokenizer()
      tok.train(text, num_merges=20)
      assert tok.decode(tok.encode(text)) == text
  ```

- [ ] **Step 2: Rodar para confirmar FAIL**

  ```bash
  pytest tests/test_tokenizer.py -v
  ```
  Expected: ImportError ou 5 FAILs.

- [ ] **Step 3: Implementar `Tokenizer` em `src/tokenizer.py`**

  - `vocab`: dict `{int: str}` — começa mapeando cada char único do texto para um ID
  - `merges`: list de `(int, int)` na ordem em que foram aprendidos
  - `train`: conta pares adjacentes, merge mais frequente → novo token, repete `num_merges` vezes
  - `encode`: converte chars em IDs, aplica merges na ordem; use lista para permitir substituições in-place
  - `decode`: join de `vocab[id]` para cada id
  - `save`/`load`: JSON com `{"vocab": {str(id): str, ...}, "merges": [[a,b], ...]}`

- [ ] **Step 4: Rodar para confirmar PASS**

  ```bash
  pytest tests/test_tokenizer.py -v
  ```
  Expected: 5 PASSes.

- [ ] **Step 5: Smoke test de velocidade com texto real (não é teste pytest)**

  ```bash
  python -c "
  from src.tokenizer import Tokenizer
  text = open('data/machado_assis.txt').read() if __import__('os').path.exists('data/machado_assis.txt') else 'a'*100000
  tok = Tokenizer(); tok.train(text[:200000], 100); print('ok', tok.vocab_size)
  "
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add src/tokenizer.py tests/test_tokenizer.py
  git commit -m "feat: BPE tokenizer with encode/decode/save/load"
  ```

---

## Task 3: Dataset Sampler

**Files:**
- Create: `src/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: numpy
- Produces:
  ```python
  def get_batch(
      token_ids: np.ndarray,   # shape (N,), dtype int32
      batch_size: int,
      context_length: int,
      rng: np.random.Generator
  ) -> tuple[np.ndarray, np.ndarray]:  # x (B,T), targets (B,T)
  ```

- [ ] **Step 1: Escrever testes**

  ```python
  # tests/test_dataset.py
  import numpy as np
  from src.dataset import get_batch

  IDS = np.arange(1000, dtype=np.int32)
  RNG = np.random.default_rng(42)

  def test_output_shapes():
      x, t = get_batch(IDS, batch_size=4, context_length=8, rng=RNG)
      assert x.shape == (4, 8)
      assert t.shape == (4, 8)

  def test_targets_are_x_shifted_right():
      x, t = get_batch(IDS, batch_size=16, context_length=8, rng=np.random.default_rng(1))
      # cada linha de t deve ser a linha de x deslocada 1 à direita
      for i in range(16):
          assert np.all(t[i] == x[i] + 1), f"row {i} failed"

  def test_dtypes_are_int():
      x, t = get_batch(IDS, batch_size=2, context_length=4, rng=RNG)
      assert x.dtype in (np.int32, np.int64)
      assert t.dtype in (np.int32, np.int64)

  def test_no_out_of_bounds():
      short = np.arange(20, dtype=np.int32)
      x, t = get_batch(short, batch_size=4, context_length=8, rng=RNG)
      assert x.max() < 20 and t.max() < 20
  ```

- [ ] **Step 2: Rodar para confirmar FAIL**

  ```bash
  pytest tests/test_dataset.py -v
  ```

- [ ] **Step 3: Implementar `get_batch` em `src/dataset.py`**

  Sorteia `batch_size` índices de início no intervalo `[0, N - context_length - 1)`, retorna janelas de tamanho `context_length` como x e as janelas deslocadas em 1 como targets.

- [ ] **Step 4: Rodar para confirmar PASS**

  ```bash
  pytest tests/test_dataset.py -v
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add src/dataset.py tests/test_dataset.py
  git commit -m "feat: dataset batch sampler"
  ```

---

## Task 4: Camadas Primitivas (Linear, LayerNorm, GELU, Embedding)

**Files:**
- Create: `src/gpt.py` (início — só as primitivas)
- Create: `tests/test_layers.py` (início)

**Interfaces:**
- Produces (todas em `src/gpt.py`):
  ```python
  class Linear:
      W: np.ndarray  # (in_features, out_features)
      b: np.ndarray  # (out_features,)
      dW: np.ndarray; db: np.ndarray
      def __init__(self, in_features: int, out_features: int, bias: bool = True)
      def forward(self, x: np.ndarray) -> np.ndarray
      def backward(self, dout: np.ndarray) -> np.ndarray  # retorna dx, acumula dW/db
      def zero_grad(self) -> None
      def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]

  class LayerNorm:
      gamma: np.ndarray; beta: np.ndarray
      dgamma: np.ndarray; dbeta: np.ndarray
      def __init__(self, d_model: int, eps: float = 1e-5)
      def forward(self, x: np.ndarray) -> np.ndarray
      def backward(self, dout: np.ndarray) -> np.ndarray
      def zero_grad(self) -> None
      def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]

  class GELU:
      def forward(self, x: np.ndarray) -> np.ndarray
      def backward(self, dout: np.ndarray) -> np.ndarray
      # sem params

  class Embedding:
      W: np.ndarray  # (num_embeddings, embedding_dim)
      dW: np.ndarray
      def __init__(self, num_embeddings: int, embedding_dim: int)
      def forward(self, idx: np.ndarray) -> np.ndarray  # idx: (B,T) int → (B,T,D)
      def backward(self, dout: np.ndarray, idx: np.ndarray) -> None  # scatter em dW
      def zero_grad(self) -> None
      def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]
  ```

- [ ] **Step 1: Escrever testes de shape e gradiente**

  ```python
  # tests/test_layers.py
  import numpy as np, pytest
  from src.gpt import Linear, LayerNorm, GELU, Embedding

  def numerical_grad(f, x, eps=1e-4):
      g = np.zeros_like(x)
      for idx in np.ndindex(*x.shape):
          orig = x[idx]; x[idx] = orig+eps; fp = f()
          x[idx] = orig-eps; fm = f(); x[idx] = orig
          g[idx] = (fp - fm) / (2*eps)
      return g

  # --- Linear ---
  def test_linear_forward_shape():
      l = Linear(4, 8)
      assert l.forward(np.ones((2, 3, 4))).shape == (2, 3, 8)

  def test_linear_backward_grad_check():
      np.random.seed(0)
      l = Linear(3, 5)
      x = np.random.randn(2, 3)
      loss = lambda: l.forward(x.copy()).sum()
      l.forward(x); l.backward(np.ones((2, 5)))
      num = numerical_grad(loss, x)
      l2 = Linear(3, 5); l2.W[:] = l.W; l2.b[:] = l.b
      l2.forward(x); dx = l2.backward(np.ones((2, 5)))
      assert np.allclose(dx, num, atol=1e-3)

  # --- LayerNorm ---
  def test_layernorm_forward_shape():
      ln = LayerNorm(8)
      out = ln.forward(np.random.randn(2, 4, 8))
      assert out.shape == (2, 4, 8)

  def test_layernorm_forward_zero_mean_unit_var():
      ln = LayerNorm(16)
      x = np.random.randn(4, 16) * 10 + 5
      out = ln.forward(x)
      assert np.allclose(out.mean(axis=-1), 0, atol=1e-5)
      assert np.allclose(out.var(axis=-1), 1, atol=1e-4)

  def test_layernorm_backward_grad_check():
      np.random.seed(1)
      ln = LayerNorm(4)
      x = np.random.randn(2, 4)
      loss = lambda: ln.forward(x.copy()).sum()
      ln.forward(x); dx = ln.backward(np.ones((2, 4)))
      num = numerical_grad(loss, x)
      assert np.allclose(dx, num, atol=1e-3)

  # --- GELU ---
  def test_gelu_shape():
      g = GELU()
      assert g.forward(np.zeros((3, 4))).shape == (3, 4)

  def test_gelu_backward_grad_check():
      np.random.seed(2)
      g = GELU()
      x = np.random.randn(2, 3)
      loss = lambda: g.forward(x.copy()).sum()
      g.forward(x); dx = g.backward(np.ones((2, 3)))
      num = numerical_grad(loss, x)
      assert np.allclose(dx, num, atol=1e-3)

  # --- NaN stability ---
  def test_layernorm_stable_large_input():
      ln = LayerNorm(8)
      x = np.ones((2, 8)) * 100
      out = ln.forward(x)
      assert not np.any(np.isnan(out))

  # --- Embedding ---
  def test_embedding_shape():
      e = Embedding(50, 16)
      idx = np.array([[1, 2, 3], [4, 5, 6]])
      assert e.forward(idx).shape == (2, 3, 16)

  def test_embedding_backward_accumulates():
      e = Embedding(10, 4)
      idx = np.array([[0, 1, 0]])  # idx 0 aparece 2x
      out = e.forward(idx)
      e.backward(np.ones_like(out), idx)
      # dW[0] deve ser soma de 2 gradientes
      assert np.allclose(e.dW[0], 2.0)
      assert np.allclose(e.dW[1], 1.0)
  ```

- [ ] **Step 2: Rodar para confirmar FAIL**

  ```bash
  pytest tests/test_layers.py -v
  ```

- [ ] **Step 3: Implementar as 4 classes em `src/gpt.py`**

  Notas de implementação:
  - `Linear.forward`: `y = x @ W + b`; guarda `self._cache = x`
  - `Linear.backward`: `dx = dout @ W.T`; `self.dW += self._cache.reshape(-1, W.shape[0]).T @ dout.reshape(-1, W.shape[1])`
  - `LayerNorm.forward`: `x_norm = (x - mean) / sqrt(var + eps)`; guarda mean, var, x_norm, x
  - `LayerNorm.backward`: fórmula padrão de LN backward (3 termos)
  - `GELU.forward`: `0.5*x*(1 + tanh(sqrt(2/π)*(x + 0.044715*x**3)))`; guarda x
  - `GELU.backward`: derivada da approx tanh via chain rule
  - `Embedding.backward`: `np.add.at(self.dW, idx, dout)`

- [ ] **Step 4: Rodar para confirmar PASS**

  ```bash
  pytest tests/test_layers.py -v
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add src/gpt.py tests/test_layers.py
  git commit -m "feat: primitive layers with forward/backward (Linear, LayerNorm, GELU, Embedding)"
  ```

---

## Task 5: CausalSelfAttention, FeedForward, TransformerBlock

**Files:**
- Modify: `src/gpt.py` (adicionar classes)
- Modify: `tests/test_layers.py` (adicionar testes)

**Interfaces:**
- Consumes: `Linear`, `LayerNorm`, `GELU` de Task 4
- Produces:
  ```python
  class CausalSelfAttention:
      def __init__(self, d_model: int, n_heads: int, context_length: int)
      def forward(self, x: np.ndarray) -> np.ndarray  # (B,T,D) → (B,T,D)
      def backward(self, dout: np.ndarray) -> np.ndarray
      def zero_grad(self) -> None
      def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]

  class FeedForward:
      def __init__(self, d_model: int, d_ff: int)
      # mesma interface

  class TransformerBlock:
      def __init__(self, d_model: int, n_heads: int, context_length: int, d_ff: int)
      # mesma interface
  ```

- [ ] **Step 1: Adicionar testes de atenção e bloco**

  ```python
  # adicionar em tests/test_layers.py
  from src.gpt import CausalSelfAttention, FeedForward, TransformerBlock

  B, T, D, H = 2, 8, 16, 2

  def test_attention_output_shape():
      attn = CausalSelfAttention(D, H, T)
      assert attn.forward(np.random.randn(B, T, D)).shape == (B, T, D)

  def test_attention_causal_mask():
      # posição i não deve depender de posição j > i
      # verificação: perturba x[:,j+1:,:] e checa que output[:,j,:] não muda
      attn = CausalSelfAttention(D, H, T)
      x = np.random.randn(1, T, D)
      out1 = attn.forward(x.copy())
      x2 = x.copy(); x2[:, 4:, :] += 10.0
      out2 = attn.forward(x2)
      assert np.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)

  def test_attention_backward_shape():
      attn = CausalSelfAttention(D, H, T)
      x = np.random.randn(B, T, D)
      attn.forward(x)
      dx = attn.backward(np.ones((B, T, D)))
      assert dx.shape == (B, T, D)

  def test_attention_backward_grad_check():
      np.random.seed(3)
      attn = CausalSelfAttention(4, 2, 4)
      x = np.random.randn(1, 4, 4) * 0.1
      loss = lambda: attn.forward(x.copy()).sum()
      attn.forward(x); dx = attn.backward(np.ones((1, 4, 4)))
      num = numerical_grad(loss, x)
      assert np.allclose(dx, num, atol=1e-2)

  def test_feedforward_shape():
      ff = FeedForward(D, 32)
      assert ff.forward(np.random.randn(B, T, D)).shape == (B, T, D)

  def test_transformer_block_shape():
      blk = TransformerBlock(D, H, T, 32)
      assert blk.forward(np.random.randn(B, T, D)).shape == (B, T, D)

  def test_transformer_block_residual():
      # output deve diferir do input (residual não é identidade com pesos não-zero)
      blk = TransformerBlock(D, H, T, 32)
      x = np.random.randn(B, T, D)
      out = blk.forward(x)
      assert not np.allclose(out, x)
  ```

- [ ] **Step 2: Rodar para confirmar FAIL**

  ```bash
  pytest tests/test_layers.py -k "attention or feedforward or transformer_block" -v
  ```

- [ ] **Step 3: Implementar `CausalSelfAttention` em `src/gpt.py`**

  - Divide `d_model` em `n_heads` cabeças de `head_dim = d_model // n_heads`
  - `forward`: projeta Q/K/V; reshape para `(B, H, T, head_dim)`; `scores = Q@K.T / sqrt(head_dim)`; aplica máscara triangular superior com `-1e9`; softmax; `out = weights @ V`; reshape e projeta Wo
  - `backward`: propaga por Wo → reshape → V/weights → softmax → scores → Q/K/V → Wq/Wk/Wv
  - Guarda em cache: x_input, Q, K, V, scores_masked, attn_weights

- [ ] **Step 4: Implementar `FeedForward` e `TransformerBlock`**

  - `FeedForward`: `Linear(d_model, d_ff)` → `GELU` → `Linear(d_ff, d_model)`
  - `TransformerBlock`: `x + attn(ln1(x))` + `x + ff(ln2(x))` (pre-LN); backward propaga pelos resíduos

- [ ] **Step 5: Rodar para confirmar PASS**

  ```bash
  pytest tests/test_layers.py -v
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add src/gpt.py tests/test_layers.py
  git commit -m "feat: CausalSelfAttention, FeedForward, TransformerBlock"
  ```

---

## Task 6: GPT Completo

**Files:**
- Modify: `src/gpt.py` (adicionar classe `GPT`)
- Create: `tests/test_gpt.py`

**Interfaces:**
- Consumes: `Embedding`, `TransformerBlock`, `LayerNorm` de Tasks 4-5
- Produces:
  ```python
  class GPT:
      def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                   n_blocks: int, context_length: int, d_ff: int)
      def forward(self, x: np.ndarray,
                  targets: np.ndarray | None = None
                  ) -> tuple[np.ndarray, float | None]
                  # logits (B,T,V), loss ou None
      def backward(self) -> None
      def zero_grad(self) -> None
      def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]
      def save(self, path: str) -> None
      @classmethod
      def load(cls, path: str, **kwargs) -> "GPT"
  ```

- [ ] **Step 1: Escrever testes**

  ```python
  # tests/test_gpt.py
  import numpy as np
  from src.gpt import GPT

  CFG = dict(vocab_size=50, d_model=16, n_heads=2,
             n_blocks=2, context_length=8, d_ff=32)

  def test_forward_logits_shape():
      m = GPT(**CFG)
      x = np.random.randint(0, 50, (2, 8))
      logits, loss = m.forward(x)
      assert logits.shape == (2, 8, 50)
      assert loss is None

  def test_forward_with_targets_returns_loss():
      m = GPT(**CFG)
      x = np.random.randint(0, 50, (2, 8))
      t = np.random.randint(0, 50, (2, 8))
      _, loss = m.forward(x, t)
      assert isinstance(loss, float)
      assert loss > 0

  def test_backward_populates_grads():
      m = GPT(**CFG)
      x = np.random.randint(0, 50, (2, 8))
      t = np.random.randint(0, 50, (2, 8))
      m.forward(x, t); m.backward()
      for param, grad in m.params_and_grads():
          assert grad is not None
          assert grad.shape == param.shape

  def test_zero_grad_clears_grads():
      m = GPT(**CFG)
      x = np.random.randint(0, 50, (2, 8))
      t = np.random.randint(0, 50, (2, 8))
      m.forward(x, t); m.backward(); m.zero_grad()
      for _, grad in m.params_and_grads():
          assert np.all(grad == 0)

  def test_weight_tying():
      # embedding W e head de saída devem ser o mesmo objeto
      m = GPT(**CFG)
      assert m.token_emb.W is m._head_W  # ou equivalente

  def test_save_load(tmp_path):
      m = GPT(**CFG)
      x = np.random.randint(0, 50, (1, 8))
      logits1, _ = m.forward(x)
      path = str(tmp_path / "model.npz")
      m.save(path)
      m2 = GPT.load(path, **CFG)
      logits2, _ = m2.forward(x)
      assert np.allclose(logits1, logits2)

  def test_loss_decreases_on_tiny_dataset():
      # 10 steps de treino em batch minúsculo devem diminuir o loss
      import sys; sys.path.insert(0, ".")
      from src.optimizer import AdamW, cosine_lr
      m = GPT(**CFG)
      x = np.random.randint(0, 50, (4, 8))
      t = np.random.randint(0, 50, (4, 8))
      opt = AdamW(m.params_and_grads, weight_decay=0.1,
                  beta1=0.9, beta2=0.95, eps=1e-8)
      losses = []
      for step in range(10):
          _, loss = m.forward(x, t)
          losses.append(loss); m.backward()
          opt.step(lr=0.001, params_and_grads=m.params_and_grads())
          m.zero_grad()
      assert losses[-1] < losses[0]
  ```

- [ ] **Step 2: Rodar para confirmar FAIL**

  ```bash
  pytest tests/test_gpt.py -v
  ```

- [ ] **Step 3: Implementar `GPT` em `src/gpt.py`**

  - `__init__`: cria `token_emb (Embedding)`, `pos_emb (Embedding)`, `n_blocks` `TransformerBlock`, `ln_f (LayerNorm)`; `self._head_W = self.token_emb.W` (weight tying)
  - `forward`: soma token + pos embeddings; passa pelos blocos; LN final; `logits = h @ self._head_W.T`; se targets: cross-entropy loss
  - `backward`: propaga `dlogits`; atenção ao weight tying — `dW` vem de dois caminhos, acumula em `token_emb.dW`
  - `params_and_grads`: coleta `(param, grad)` de todos os sub-módulos recursivamente; exclui `_head_W` duplicado
  - `save`: `np.savez(path, **{str(i): p for i, (p, _) in enumerate(self.params_and_grads())})`
  - `load`: carrega array por índice, copia em params na mesma ordem

- [ ] **Step 4: Rodar para confirmar PASS**

  ```bash
  pytest tests/test_gpt.py -v
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add src/gpt.py tests/test_gpt.py
  git commit -m "feat: GPT full model — forward, backward, save/load"
  ```

---

## Task 7: AdamW + Cosine Annealing

**Files:**
- Create: `src/optimizer.py`
- Create: `tests/test_optimizer.py`

**Interfaces:**
- Produces:
  ```python
  def cosine_lr(step: int, total_steps: int,
                lr_max: float, lr_min: float) -> float

  class AdamW:
      def __init__(self, params_and_grads_fn: Callable,
                   weight_decay: float, beta1: float,
                   beta2: float, eps: float)
      def step(self, lr: float,
               params_and_grads: list[tuple[np.ndarray, np.ndarray]]
               ) -> None
  ```

- [ ] **Step 1: Escrever testes**

  ```python
  # tests/test_optimizer.py
  import numpy as np, pytest
  from src.optimizer import AdamW, cosine_lr

  def test_cosine_lr_endpoints():
      assert cosine_lr(0, 100, 0.001, 0.0001) == pytest.approx(0.001, rel=1e-5)
      assert cosine_lr(100, 100, 0.001, 0.0001) == pytest.approx(0.0001, rel=1e-4)

  def test_cosine_lr_monotone_decreasing():
      lrs = [cosine_lr(s, 100, 0.001, 0.0001) for s in range(101)]
      assert all(lrs[i] >= lrs[i+1] for i in range(100))

  def test_adamw_updates_params():
      p = np.array([1.0, 2.0, 3.0])
      g = np.array([0.1, 0.1, 0.1])
      p_before = p.copy()
      opt = AdamW(lambda: [(p, g)], weight_decay=0.0,
                  beta1=0.9, beta2=0.95, eps=1e-8)
      opt.step(lr=0.001, params_and_grads=[(p, g)])
      assert not np.allclose(p, p_before)

  def test_adamw_weight_decay_shrinks_params():
      p = np.array([1.0, 1.0])
      g = np.zeros(2)
      opt = AdamW(lambda: [(p, g)], weight_decay=0.1,
                  beta1=0.9, beta2=0.95, eps=1e-8)
      opt.step(lr=0.001, params_and_grads=[(p, g)])
      assert np.all(p < 1.0)

  def test_adamw_bias_correction():
      # no primeiro step o lr efetivo deve ser próximo do lr passado
      p = np.ones(4)
      g = np.ones(4)
      opt = AdamW(lambda: [(p, g)], weight_decay=0.0,
                  beta1=0.9, beta2=0.95, eps=1e-8)
      p_before = p.copy()
      opt.step(lr=0.001, params_and_grads=[(p, g)])
      delta = np.abs(p - p_before)
      assert np.all(delta < 0.01)  # não explode no step 1
  ```

- [ ] **Step 2: Rodar para confirmar FAIL**

  ```bash
  pytest tests/test_optimizer.py -v
  ```

- [ ] **Step 3: Implementar `cosine_lr` e `AdamW` em `src/optimizer.py`**

  - `cosine_lr`: `lr_min + 0.5*(lr_max-lr_min)*(1 + cos(π*step/total_steps))`
  - `AdamW.__init__`: `self.t=0`, `self.m={}`, `self.v={}` (inicializa lazy no primeiro step)
  - `AdamW.step`: usa `id(param)` como chave nos dicts; aplica weight decay antes do update Adam; bias-correct m e v

- [ ] **Step 4: Rodar para confirmar PASS**

  ```bash
  pytest tests/test_optimizer.py -v
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add src/optimizer.py tests/test_optimizer.py
  git commit -m "feat: AdamW optimizer with cosine LR annealing"
  ```

---

## Task 8: Loop de Treino

**Files:**
- Create: `train.py`

**Interfaces:**
- Consumes: `config.*`, `Tokenizer`, `get_batch`, `GPT`, `AdamW`, `cosine_lr`
- Produces: `checkpoints/epoch_XX.npz`, logs de loss no stdout

- [ ] **Step 1: Verificar que o corpus pode ser baixado**

  ```bash
  python -c "
  import urllib.request, os
  os.makedirs('data', exist_ok=True)
  url = 'https://www.gutenberg.org/cache/epub/55752/pg55752.txt'
  urllib.request.urlretrieve(url, 'data/machado_assis.txt')
  text = open('data/machado_assis.txt', encoding='utf-8', errors='ignore').read()
  print(f'Corpus: {len(text):,} chars')
  "
  ```
  Expected: `Corpus: N chars` (N > 300 000).

- [ ] **Step 2: Implementar `train.py`**

  Fluxo exato (sem testes automatizados — verificação é rodar):
  ```
  1. import config.*
  2. Baixar corpus se não existir (urllib.request)
  3. Carregar texto, limpar header/footer do Gutenberg
  4. Se TOKENIZER_FILE existir: Tokenizer.load(); senão: train(text, num_merges=10_000), save()
  5. token_ids = np.array(tok.encode(text), dtype=np.int32)
  6. split 90/10 → train_ids, val_ids
  7. model = GPT(tok.vocab_size, EMBEDDING_DIM, N_HEADS, N_BLOCKS, SEQUENCE_LENGTH, d_ff=EMBEDDING_DIM*4)
  8. Se existir checkpoint mais recente em CHECKPOINT_DIR: model.load() + registrar epoch_start
  9. opt = AdamW(model.params_and_grads, WEIGHT_DECAY, BETA1, BETA2, eps=1e-8)
  10. total_steps = EPOCHS * (len(train_ids) // (BATCH_SIZE * SEQUENCE_LENGTH))
  11. step = epoch_start * steps_per_epoch
  12. Para epoch em tqdm(range(epoch_start, EPOCHS)):
       a. Para cada batch na época (loop sobre train_ids em janelas):
          - lr = cosine_lr(step, total_steps, LR_MAX, LR_MIN)
          - x, t = get_batch(train_ids, BATCH_SIZE, SEQUENCE_LENGTH, rng)
          - _, loss = model.forward(x, t)
          - model.backward(); opt.step(lr, model.params_and_grads()); model.zero_grad()
          - step += 1
       b. Val loss: média de VALIDATION_BATCHES batches de val_ids (só forward)
       c. Print: f"epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | lr {lr:.6f}"
       d. Sample: generate(model, tok, SAMPLE_PROMPT, SAMPLE_TOKENS, temperature=0.8), print
       e. model.save(f"{CHECKPOINT_DIR}/epoch_{epoch:02d}.npz")
  ```

- [ ] **Step 3: Smoke test — 1 época em dataset reduzido**

  ```bash
  python -c "
  import os; os.environ['ANTON_SMOKE'] = '1'
  " && python train.py
  ```

  Ou editar train.py temporariamente para `EPOCHS=1`, `BATCH_SIZE=4`. Expected: epoch 0 completa sem erro, checkpoint salvo.

- [ ] **Step 4: Commit**

  ```bash
  git add train.py data/machado_assis.txt data/tokenizer.json
  git commit -m "feat: training loop with cosine LR, checkpointing, sampling"
  ```

---

## Task 9: Geração

**Files:**
- Create: `src/generate.py`
- Create: `tests/test_generate.py`

**Interfaces:**
- Consumes: `GPT`, `Tokenizer`, `config.*`
- Produces:
  ```python
  def generate(
      model: GPT,
      tokenizer: Tokenizer,
      prompt: str,
      max_new_tokens: int = 200,
      temperature: float = 1.0
  ) -> str
  ```
  CLI: `python src/generate.py --checkpoint <path> --prompt <str> --tokens <int> --temp <float>`

- [ ] **Step 1: Escrever testes**

  ```python
  # tests/test_generate.py
  import numpy as np, pytest
  from src.gpt import GPT
  from src.tokenizer import Tokenizer
  from src.generate import generate

  CFG = dict(vocab_size=50, d_model=16, n_heads=2,
             n_blocks=2, context_length=8, d_ff=32)

  def test_generate_longer_than_prompt():
      tok = Tokenizer()
      tok.train("abcde " * 500, num_merges=20)
      m = GPT(**{**CFG, 'vocab_size': tok.vocab_size})
      out = generate(m, tok, "abc", max_new_tokens=10, temperature=1.0)
      assert len(out) > len("abc")

  def test_generate_returns_string():
      tok = Tokenizer()
      tok.train("hello world " * 500, num_merges=20)
      m = GPT(**{**CFG, 'vocab_size': tok.vocab_size})
      out = generate(m, tok, "hello", max_new_tokens=5, temperature=1.0)
      assert isinstance(out, str)

  def test_generate_low_temperature_deterministic():
      tok = Tokenizer()
      tok.train("abcde " * 500, num_merges=20)
      m = GPT(**{**CFG, 'vocab_size': tok.vocab_size})
      o1 = generate(m, tok, "abc", max_new_tokens=10, temperature=0.01)
      o2 = generate(m, tok, "abc", max_new_tokens=10, temperature=0.01)
      assert o1 == o2  # temperatura muito baixa → quase determinístico
  ```

- [ ] **Step 2: Rodar para confirmar FAIL**

  ```bash
  pytest tests/test_generate.py -v
  ```

- [ ] **Step 3: Implementar `generate` e CLI em `src/generate.py`**

  - `generate`: encode prompt; loop `max_new_tokens` vezes: trim contexto para `context_length`, forward sem targets, pega logits da última posição, divide por `temperature`, softmax → probs, `np.random.choice(vocab_size, p=probs)`, append; decode IDs
  - CLI: `argparse` com `--checkpoint`, `--prompt`, `--tokens` (default 200), `--temp` (default 0.8); carrega model + tokenizer; imprime resultado

- [ ] **Step 4: Rodar para confirmar PASS**

  ```bash
  pytest tests/test_generate.py -v
  ```

- [ ] **Step 5: Teste de integração manual com checkpoint real (se disponível)**

  ```bash
  python src/generate.py --checkpoint checkpoints/epoch_00.npz \
      --prompt "— Mas, senhora" --tokens 200 --temp 0.8
  ```

- [ ] **Step 6: Commit final**

  ```bash
  git add src/generate.py tests/test_generate.py
  git commit -m "feat: autoregressive generation with temperature + CLI"
  ```

---

## Sumário das Tasks

| # | Componente | Arquivo principal | Testes |
|---|---|---|---|
| 1 | Scaffold | `config.py` | — |
| 2 | BPE Tokenizer | `src/tokenizer.py` | `tests/test_tokenizer.py` |
| 3 | Dataset sampler | `src/dataset.py` | `tests/test_dataset.py` |
| 4 | Camadas primitivas | `src/gpt.py` | `tests/test_layers.py` |
| 5 | Attention + Block | `src/gpt.py` | `tests/test_layers.py` |
| 6 | GPT completo | `src/gpt.py` | `tests/test_gpt.py` |
| 7 | AdamW + cosine LR | `src/optimizer.py` | `tests/test_optimizer.py` |
| 8 | Training loop | `train.py` | smoke test manual |
| 9 | Geração | `src/generate.py` | `tests/test_generate.py` |
