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
