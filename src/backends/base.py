from typing import Protocol

from src.training import RunOptions, TrainConfig


class Backend(Protocol):
    name: str

    def run(self, settings: TrainConfig, options: RunOptions) -> str | None:
        """Executa o treino e retorna o caminho do último checkpoint (ou None)."""
        ...
