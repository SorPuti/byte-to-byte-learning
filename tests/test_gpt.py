import numpy as np
from src.gpt import GPT

CFG = dict(vocab_size=50, d_model=16, n_heads=2, n_blocks=2, context_length=8, d_ff=32)


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
    m.forward(x, t)
    m.backward()
    for param, grad in m.params_and_grads():
        assert grad is not None
        assert grad.shape == param.shape


def test_zero_grad_clears_grads():
    m = GPT(**CFG)
    x = np.random.randint(0, 50, (2, 8))
    t = np.random.randint(0, 50, (2, 8))
    m.forward(x, t)
    m.backward()
    m.zero_grad()
    for _, grad in m.params_and_grads():
        assert np.all(grad == 0)


def test_weight_tying():
    m = GPT(**CFG)
    assert m.token_emb.W is m._head_W


def test_tied_embedding_grad_matches_numerical():
    np.random.seed(7)
    m = GPT(**CFG)
    x = np.random.randint(0, 50, (1, 8))
    t = np.random.randint(0, 50, (1, 8))
    _, loss = m.forward(x, t)
    m.backward()
    W = m.token_emb.W
    rows = [int(x[0, 0]), int(t[0, 3]), 49]
    eps = 1e-3
    for r in rows:
        for c in (0, 5):
            orig = W[r, c]
            W[r, c] = orig + eps
            lp = m.forward(x, t)[1]
            W[r, c] = orig - eps
            lm = m.forward(x, t)[1]
            W[r, c] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(m.token_emb.dW[r, c] - num) < 1e-3


def test_save_load(tmp_path):
    m = GPT(**CFG)
    x = np.random.randint(0, 50, (1, 8))
    logits1, _ = m.forward(x)
    path = str(tmp_path / "model.npz")
    m.save(path)
    m2 = GPT.load(path, **CFG)
    logits2, _ = m2.forward(x)
    assert np.allclose(logits1, logits2)
    assert m2.token_emb.W is m2._head_W
