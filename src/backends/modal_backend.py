from dataclasses import asdict

from src.training import RunOptions, TrainConfig


class ModalBackend:
    """Envia o treino para o Modal. Importa `modal_app` só na execução.

    Os caminhos são resolvidos dentro do container (Volumes), por isso não são enviados.
    """

    name = "modal"

    def run(self, settings: TrainConfig, options: RunOptions) -> str | None:
        from modal_app import app, train_remote  # import tardio: `modal` só é exigido aqui

        with app.run():
            return train_remote.remote(asdict(settings), asdict(options))
