import math

import numpy as np

_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        std = np.sqrt(2.0 / (in_features + out_features))
        self.W = (np.random.randn(in_features, out_features) * std).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros(out_features, dtype=np.float32) if bias else None
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        in_f, out_f = self.W.shape
        x2 = self._x.reshape(-1, in_f)
        d2 = dout.reshape(-1, out_f)
        self.dW += x2.T @ d2
        if self.db is not None:
            self.db += d2.sum(axis=0)
        return (d2 @ self.W.T).reshape(self._x.shape)

    def zero_grad(self) -> None:
        self.dW[:] = 0
        if self.db is not None:
            self.db[:] = 0

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        out = [(self.W, self.dW)]
        if self.b is not None:
            out.append((self.b, self.db))
        return out


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.dgamma = np.zeros(d_model, dtype=np.float32)
        self.dbeta = np.zeros(d_model, dtype=np.float32)
        self.eps = eps
        self._xhat = None
        self._inv_std = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._inv_std = 1.0 / np.sqrt(var + self.eps)
        self._xhat = (x - mu) * self._inv_std
        return self.gamma * self._xhat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        xhat = self._xhat
        d = xhat.shape[-1]
        flat_axes = tuple(range(dout.ndim - 1))
        self.dgamma += (dout * xhat).sum(axis=flat_axes)
        self.dbeta += dout.sum(axis=flat_axes)
        dxhat = dout * self.gamma
        return (
            self._inv_std
            / d
            * (
                d * dxhat
                - dxhat.sum(axis=-1, keepdims=True)
                - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
            )
        )

    def zero_grad(self) -> None:
        self.dgamma[:] = 0
        self.dbeta[:] = 0

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class GELU:
    def __init__(self):
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x**3)))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._x
        u = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
        t = np.tanh(u)
        du = _SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * x**2)
        dy = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du
        return dout * dy

    def zero_grad(self) -> None:
        pass

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return []


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.W = (np.random.randn(num_embeddings, embedding_dim) * 0.02).astype(np.float32)
        self.dW = np.zeros_like(self.W)

    def forward(self, idx: np.ndarray) -> np.ndarray:
        return self.W[idx]

    def backward(self, dout: np.ndarray, idx: np.ndarray) -> None:
        # Equivalente a np.add.at(self.dW, idx, dout), mas vetorizado: np.add.at é lento.
        flat_idx = idx.reshape(-1)
        flat = dout.reshape(-1, self.dW.shape[1])
        order = np.argsort(flat_idx, kind="stable")
        sorted_idx = flat_idx[order]
        uniq, starts = np.unique(sorted_idx, return_index=True)
        sums = np.add.reduceat(flat[order], starts, axis=0)
        self.dW[uniq] += sums.astype(self.dW.dtype, copy=False)

    def zero_grad(self) -> None:
        self.dW[:] = 0

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW)]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class CausalSelfAttention:
    def __init__(self, d_model: int, n_heads: int, context_length: int):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = Linear(d_model, d_model, bias=False)
        self.Wk = Linear(d_model, d_model, bias=False)
        self.Wv = Linear(d_model, d_model, bias=False)
        self.Wo = Linear(d_model, d_model, bias=False)
        self._mask = np.triu(np.ones((context_length, context_length), dtype=bool), k=1)
        self._q = self._k = self._v = self._weights = None

    def _split(self, x: np.ndarray) -> np.ndarray:
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge(self, x: np.ndarray) -> np.ndarray:
        b, _, t, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, self.n_heads * self.head_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        t = x.shape[1]
        self._q = self._split(self.Wq.forward(x))
        self._k = self._split(self.Wk.forward(x))
        self._v = self._split(self.Wv.forward(x))
        # math.sqrt (float Python): np.sqrt devolve float64 e promove a atenção inteira para float64.
        scores = self._q @ self._k.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        scores = np.where(self._mask[:t, :t], -1e9, scores)
        self._weights = _softmax(scores)
        out = self._weights @ self._v
        return self.Wo.forward(self._merge(out))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        do = self._split(self.Wo.backward(dout))
        w = self._weights
        dw = do @ self._v.transpose(0, 1, 3, 2)
        dv = w.transpose(0, 1, 3, 2) @ do
        dscores = w * (dw - (dw * w).sum(axis=-1, keepdims=True))
        dscores = dscores / math.sqrt(self.head_dim)
        dq = dscores @ self._k
        dk = dscores.transpose(0, 1, 3, 2) @ self._q
        return (
            self.Wq.backward(self._merge(dq))
            + self.Wk.backward(self._merge(dk))
            + self.Wv.backward(self._merge(dv))
        )

    def zero_grad(self) -> None:
        for layer in (self.Wq, self.Wk, self.Wv, self.Wo):
            layer.zero_grad()

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for layer in (self.Wq, self.Wk, self.Wv, self.Wo):
            out.extend(layer.params_and_grads())
        return out


class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.lin1 = Linear(d_model, d_ff)
        self.act = GELU()
        self.lin2 = Linear(d_ff, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.lin2.forward(self.act.forward(self.lin1.forward(x)))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.lin1.backward(self.act.backward(self.lin2.backward(dout)))

    def zero_grad(self) -> None:
        self.lin1.zero_grad()
        self.lin2.zero_grad()

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.lin1.params_and_grads() + self.lin2.params_and_grads()


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, context_length: int, d_ff: int):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x + self.attn.forward(self.ln1.forward(x))
        return h + self.ff.forward(self.ln2.forward(h))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dh = dout + self.ln2.backward(self.ff.backward(dout))
        return dh + self.ln1.backward(self.attn.backward(dh))

    def zero_grad(self) -> None:
        for m in (self.ln1, self.attn, self.ln2, self.ff):
            m.zero_grad()

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for m in (self.ln1, self.attn, self.ln2, self.ff):
            out.extend(m.params_and_grads())
        return out


class GPT:
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_blocks: int,
        context_length: int,
        d_ff: int,
    ):
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = Embedding(context_length, d_model)
        self.blocks = [
            TransformerBlock(d_model, n_heads, context_length, d_ff) for _ in range(n_blocks)
        ]
        self.ln_f = LayerNorm(d_model)
        self._head_W = self.token_emb.W
        self._cache = None

    def forward(
        self, x: np.ndarray, targets: np.ndarray | None = None
    ) -> tuple[np.ndarray, float | None]:
        t = x.shape[1]
        h = self.token_emb.forward(x) + self.pos_emb.forward(np.arange(t))
        for block in self.blocks:
            h = block.forward(h)
        h = self.ln_f.forward(h)
        logits = h @ self._head_W.T
        if targets is None:
            self._cache = None
            return logits, None
        probs = logits - logits.max(axis=-1, keepdims=True)
        np.exp(probs, out=probs)
        probs /= probs.sum(axis=-1, keepdims=True)
        loss = float(-np.log(np.take_along_axis(probs, targets[..., None], axis=-1)).mean())
        self._cache = (x, h, probs, targets)
        return logits, loss

    def backward(self) -> None:
        x, h, probs, targets = self._cache
        b, t, v = probs.shape
        dlogits = probs
        # Pares (b, t) são únicos, então indexação direta equivale a subtract.at e é mais rápida.
        dlogits[np.arange(b)[:, None], np.arange(t)[None, :], targets] -= 1.0
        dlogits /= b * t
        d_head = dlogits.reshape(-1, v).T @ h.reshape(-1, h.shape[-1])
        self.token_emb.dW += d_head
        dh = dlogits @ self._head_W
        dh = self.ln_f.backward(dh)
        for block in reversed(self.blocks):
            dh = block.backward(dh)
        self.pos_emb.backward(dh.sum(axis=0), np.arange(t))
        self.token_emb.backward(dh, x)

    def zero_grad(self) -> None:
        self.token_emb.zero_grad()
        self.pos_emb.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.ln_f.zero_grad()

    def params_and_grads(self) -> list[tuple[np.ndarray, np.ndarray]]:
        out = self.token_emb.params_and_grads() + self.pos_emb.params_and_grads()
        for block in self.blocks:
            out.extend(block.params_and_grads())
        out.extend(self.ln_f.params_and_grads())
        return out

    def save(self, path: str) -> None:
        arrays = {str(i): p for i, (p, _) in enumerate(self.params_and_grads())}
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str, **kwargs) -> "GPT":
        model = cls(**kwargs)
        data = np.load(path)
        for i, (p, _) in enumerate(model.params_and_grads()):
            p[...] = data[str(i)]
        return model


def model_kwargs(vocab_size: int) -> dict:
    import config

    return dict(
        vocab_size=vocab_size,
        d_model=config.EMBEDDING_DIM,
        n_heads=config.N_HEADS,
        n_blocks=config.N_BLOCKS,
        context_length=config.SEQUENCE_LENGTH,
        d_ff=config.D_FF,
    )
