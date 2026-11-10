import math
from typing import Callable

import numpy as np


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step / total_steps))


class AdamW:
    def __init__(
        self,
        params_and_grads_fn: Callable,
        weight_decay: float,
        beta1: float,
        beta2: float,
        eps: float,
    ):
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self, lr: float, params_and_grads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.t += 1
        bc1 = 1 - self.beta1**self.t
        bc2 = 1 - self.beta2**self.t
        for param, grad in params_and_grads:
            key = id(param)
            if key not in self._m:
                self._m[key] = np.zeros_like(param)
                self._v[key] = np.zeros_like(param)
            m, v = self._m[key], self._v[key]
            m[...] = self.beta1 * m + (1 - self.beta1) * grad
            v[...] = self.beta2 * v + (1 - self.beta2) * grad**2
            if param.ndim >= 2 and self.weight_decay:
                param -= lr * self.weight_decay * param
            param -= lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
