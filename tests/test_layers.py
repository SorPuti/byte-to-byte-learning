import numpy as np
import pytest
from src.gpt import Linear, LayerNorm, GELU, Embedding


def numerical_grad(f, x, eps=1e-4):
    g = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        fp = f()
        x[idx] = orig - eps
        fm = f()
        x[idx] = orig
        g[idx] = (fp - fm) / (2 * eps)
    return g


def test_linear_forward_shape():
    l = Linear(4, 8)
    assert l.forward(np.ones((2, 3, 4))).shape == (2, 3, 8)


def test_linear_backward_grad_check():
    np.random.seed(0)
    l = Linear(3, 5)
    x = np.random.randn(2, 3)
    loss = lambda: l.forward(x.copy()).sum()
    l.forward(x)
    dx = l.backward(np.ones((2, 5)))
    num = numerical_grad(loss, x)
    assert np.allclose(dx, num, atol=1e-3)


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
    ln.forward(x)
    dx = ln.backward(np.ones((2, 4)))
    num = numerical_grad(loss, x)
    assert np.allclose(dx, num, atol=1e-3)


def test_layernorm_stable_large_input():
    ln = LayerNorm(8)
    out = ln.forward(np.ones((2, 8)) * 100)
    assert not np.any(np.isnan(out))


def test_gelu_shape():
    assert GELU().forward(np.zeros((3, 4))).shape == (3, 4)


def test_gelu_backward_grad_check():
    np.random.seed(2)
    g = GELU()
    x = np.random.randn(2, 3)
    loss = lambda: g.forward(x.copy()).sum()
    g.forward(x)
    dx = g.backward(np.ones((2, 3)))
    num = numerical_grad(loss, x)
    assert np.allclose(dx, num, atol=1e-3)


def test_embedding_shape():
    e = Embedding(50, 16)
    idx = np.array([[1, 2, 3], [4, 5, 6]])
    assert e.forward(idx).shape == (2, 3, 16)


def test_embedding_backward_accumulates():
    e = Embedding(10, 4)
    idx = np.array([[0, 1, 0]])
    out = e.forward(idx)
    e.backward(np.ones_like(out), idx)
    assert np.allclose(e.dW[0], 2.0)
    assert np.allclose(e.dW[1], 1.0)


from src.gpt import CausalSelfAttention, FeedForward, TransformerBlock

B, T, D, H = 2, 8, 16, 2


def test_attention_output_shape():
    attn = CausalSelfAttention(D, H, T)
    assert attn.forward(np.random.randn(B, T, D)).shape == (B, T, D)


def test_attention_causal_mask_blocks_future_positions():
    attn = CausalSelfAttention(D, H, T)
    x = np.random.randn(1, T, D)
    out1 = attn.forward(x.copy())
    x2 = x.copy()
    x2[:, 4:, :] += 10.0
    out2 = attn.forward(x2)
    assert np.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)


def test_attention_weights_are_zero_above_diagonal():
    attn = CausalSelfAttention(D, H, T)
    attn.forward(np.random.randn(B, T, D))
    w = attn._weights
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    assert np.all(w[:, :, upper] == 0)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_attention_backward_shape():
    attn = CausalSelfAttention(D, H, T)
    attn.forward(np.random.randn(B, T, D))
    dx = attn.backward(np.ones((B, T, D)))
    assert dx.shape == (B, T, D)


def test_attention_backward_grad_check():
    np.random.seed(3)
    attn = CausalSelfAttention(4, 2, 4)
    x = np.random.randn(1, 4, 4) * 0.1
    loss = lambda: attn.forward(x.copy()).sum()
    attn.forward(x)
    dx = attn.backward(np.ones((1, 4, 4)))
    num = numerical_grad(loss, x)
    assert np.allclose(dx, num, atol=1e-2)


def test_feedforward_shape():
    ff = FeedForward(D, 32)
    assert ff.forward(np.random.randn(B, T, D)).shape == (B, T, D)


def test_feedforward_backward_grad_check():
    np.random.seed(4)
    ff = FeedForward(4, 6)
    x = np.random.randn(1, 3, 4)
    loss = lambda: ff.forward(x.copy()).sum()
    ff.forward(x)
    dx = ff.backward(np.ones((1, 3, 4)))
    num = numerical_grad(loss, x)
    assert np.allclose(dx, num, atol=1e-3)


def test_transformer_block_shape():
    blk = TransformerBlock(D, H, T, 32)
    assert blk.forward(np.random.randn(B, T, D)).shape == (B, T, D)


def test_transformer_block_residual():
    blk = TransformerBlock(D, H, T, 32)
    x = np.random.randn(B, T, D)
    assert not np.allclose(blk.forward(x), x)


def test_transformer_block_backward_grad_check():
    np.random.seed(5)
    blk = TransformerBlock(4, 2, 3, 6)
    x = np.random.randn(1, 3, 4) * 0.5
    loss = lambda: blk.forward(x.copy()).sum()
    blk.forward(x)
    dx = blk.backward(np.ones((1, 3, 4)))
    num = numerical_grad(loss, x)
    assert np.allclose(dx, num, atol=1e-2)
