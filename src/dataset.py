import numpy as np


def get_batch(
    token_ids: np.ndarray,
    batch_size: int,
    context_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    starts = rng.integers(0, len(token_ids) - context_length - 1, size=batch_size)
    x = np.stack([token_ids[i : i + context_length] for i in starts])
    targets = np.stack([token_ids[i + 1 : i + context_length + 1] for i in starts])
    return x, targets
