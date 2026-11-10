import numpy as np
import pytest
from src.gpt import GPT
from src.optimizer import AdamW, cosine_lr

CFG = dict(vocab_size=50, d_model=16, n_heads=2, n_blocks=2, context_length=8, d_ff=32)


def test_cosine_lr_endpoints():
    assert cosine_lr(0, 100, 0.001, 0.0001) == pytest.approx(0.001, rel=1e-5)
    assert cosine_lr(100, 100, 0.001, 0.0001) == pytest.approx(0.0001, rel=1e-4)


def test_cosine_lr_monotone_decreasing():
    lrs = [cosine_lr(s, 100, 0.001, 0.0001) for s in range(101)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(100))


def test_adamw_updates_params():
    p = np.array([1.0, 2.0, 3.0])
    g = np.array([0.1, 0.1, 0.1])
    p_before = p.copy()
    opt = AdamW(lambda: [(p, g)], weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    opt.step(lr=0.001, params_and_grads=[(p, g)])
    assert not np.allclose(p, p_before)


def test_adamw_weight_decay_shrinks_matrices():
    p = np.ones((2, 2))
    g = np.zeros((2, 2))
    opt = AdamW(lambda: [(p, g)], weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8)
    opt.step(lr=0.001, params_and_grads=[(p, g)])
    assert np.all(p < 1.0)


def test_adamw_skips_weight_decay_on_vectors():
    p = np.ones(4)
    g = np.zeros(4)
    opt = AdamW(lambda: [(p, g)], weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8)
    opt.step(lr=0.001, params_and_grads=[(p, g)])
    assert np.allclose(p, 1.0)


def test_adamw_bias_correction():
    p = np.ones(4)
    g = np.ones(4)
    opt = AdamW(lambda: [(p, g)], weight_decay=0.0, beta1=0.9, beta2=0.95, eps=1e-8)
    p_before = p.copy()
    opt.step(lr=0.001, params_and_grads=[(p, g)])
    assert np.all(np.abs(p - p_before) < 0.01)


def test_loss_decreases_on_tiny_dataset():
    np.random.seed(0)
    m = GPT(**CFG)
    x = np.random.randint(0, 50, (4, 8))
    t = np.random.randint(0, 50, (4, 8))
    opt = AdamW(m.params_and_grads, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8)
    losses = []
    for _ in range(10):
        _, loss = m.forward(x, t)
        losses.append(loss)
        m.backward()
        opt.step(lr=0.001, params_and_grads=m.params_and_grads())
        m.zero_grad()
    assert losses[-1] < losses[0]
