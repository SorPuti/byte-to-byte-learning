"""Backends de execução do treino. Cada um implementa `Backend.run`."""

from src.backends.base import Backend
from src.backends.local import LocalBackend

__all__ = ["Backend", "LocalBackend"]
