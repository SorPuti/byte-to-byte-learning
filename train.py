import argparse

import config
from prepare_data import ensure_corpus
from src.backends import LocalBackend
from src.backends.base import Backend
from src.training import Paths, RunOptions, TrainConfig


def build_backend(name: str, checkpoint_dir: str) -> Backend:
    if name == "local":
        paths = Paths(
            corpus_file=config.CORPUS_FILE,
            tokenizer_file=config.TOKENIZER_FILE,
            checkpoint_dir=checkpoint_dir,
        )
        return LocalBackend(paths, ensure_corpus=ensure_corpus)
    if name == "modal":
        from src.backends.modal_backend import ModalBackend  # import tardio: só exige `modal` aqui

        return ModalBackend()
    raise ValueError(f"backend desconhecido: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo byte-to-byte em português.")
    parser.add_argument("--backend", choices=["local", "modal"], default="local", help="onde executar o treino")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="número de épocas a executar")
    parser.add_argument("--checkpoint-dir", default=config.CHECKPOINT_DIR, help="só usado no backend local")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="teste rápido: 5 passos por época e 1 batch de validação",
    )
    args = parser.parse_args()

    settings = TrainConfig.from_module(config)
    options = RunOptions(epochs=args.epochs, smoke=args.smoke)
    backend = build_backend(args.backend, args.checkpoint_dir)
    backend.run(settings, options)


if __name__ == "__main__":
    main()
