"""Entrypoint do treino no Modal.

Uso (executado por você, não automaticamente):
    modal run modal_app.py --epochs 3
    modal run modal_app.py --epochs 1 --smoke

Pré-requisito: subir o corpus para o Volume de dados uma vez:
    modal volume put byte-to-byte-data data/corpus_pt.txt /corpus_pt.txt

Os checkpoints ficam no Volume `byte-to-byte-checkpoints`.
Baixar um checkpoint: modal volume get byte-to-byte-checkpoints epoch_74.npz ./checkpoints/
"""

from dataclasses import asdict

import modal

import config
from src.training import RunOptions, TrainConfig

APP_NAME = "byte-to-byte-train"
DATA_VOLUME_NAME = "byte-to-byte-data"
CHECKPOINT_VOLUME_NAME = "byte-to-byte-checkpoints"

# Python da imagem remota. O projeto pede 3.14 (.python-version): confirme que o
# Modal suporta essa versão; se não, troque aqui por uma versão suportada.
PYTHON_VERSION = "3.14"

# Recursos: o modelo é NumPy puro em CPU, então o que acelera é número de vCPUs.
CPU_CORES = 8.0
MEMORY_MB = 16 * 1024
TIMEOUT_SECONDS = 12 * 60 * 60

DATA_MOUNT = "/data"
CHECKPOINT_MOUNT = "/checkpoints"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("src", "config")
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={DATA_MOUNT: data_volume, CHECKPOINT_MOUNT: checkpoint_volume},
)
def train_remote(settings: dict, options: dict) -> str | None:
    from src.training import Paths, RunOptions, TrainConfig, train

    paths = Paths(
        corpus_file=f"{DATA_MOUNT}/corpus_pt.txt",
        tokenizer_file=f"{DATA_MOUNT}/tokenizer.json",
        checkpoint_dir=CHECKPOINT_MOUNT,
    )
    try:
        return train(
            TrainConfig(**settings),
            paths,
            RunOptions(**options),
            on_save=lambda _path: checkpoint_volume.commit(),
        )
    finally:
        # Tokenizer criado na primeira execução precisa persistir no Volume de dados.
        data_volume.commit()


@app.local_entrypoint()
def main(epochs: int = config.EPOCHS, smoke: bool = False) -> None:
    # Já estamos dentro do contexto do `modal run`, então chamamos a função direto
    # (ModalBackend abriria um app.run() aninhado).
    settings = TrainConfig.from_module(config)
    options = RunOptions(epochs=epochs, smoke=smoke)
    last = train_remote.remote(asdict(settings), asdict(options))
    print(f"Último checkpoint: {last}")
