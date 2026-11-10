import numpy as np

from src.gpt import GPT, Embedding


def test_embedding_backward_matches_add_at_reference():
    rng = np.random.default_rng(0)
    e = Embedding(50, 8)
    idx = rng.integers(0, 50, size=(4, 6))
    dout = rng.standard_normal((4, 6, 8)).astype(np.float32)
    e.backward(dout, idx)
    ref = np.zeros((50, 8), dtype=np.float32)
    np.add.at(ref, idx, dout)
    np.testing.assert_allclose(e.dW, ref, rtol=1e-5, atol=1e-5)


def test_embedding_backward_accumulates_into_existing_grad():
    e = Embedding(5, 2)
    idx = np.array([1, 1, 3])
    dout = np.ones((3, 2), dtype=np.float32)
    e.backward(dout, idx)
    e.backward(dout, idx)
    assert e.dW[1].tolist() == [4.0, 4.0]
    assert e.dW[3].tolist() == [2.0, 2.0]
    assert e.dW[0].tolist() == [0.0, 0.0]


def test_attention_and_gradients_stay_float32():
    model = GPT(vocab_size=50, d_model=16, n_heads=2, n_blocks=1, context_length=8, d_ff=32)
    x = np.random.default_rng(1).integers(0, 50, size=(2, 8))
    _, loss = model.forward(x, x)
    assert model.blocks[0].attn._weights.dtype == np.float32
    model.backward()
    assert all(g.dtype == np.float32 for _, g in model.params_and_grads())
    assert np.isfinite(loss)
