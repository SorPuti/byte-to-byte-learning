"""Loop de treino independente de ambiente.

Não importa `config` nem `prepare_data`: todos os parâmetros entram explicitamente
por `TrainConfig`, `Paths` e `RunOptions`. Assim o mesmo código roda localmente
ou dentro de um container remoto (Modal) sem alterações.
"""

import glob
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import numpy as np
from tqdm import tqdm

from src.dataset import get_batch
from src.generate import generate
from src.gpt import GPT
from src.optimizer import AdamW, cosine_lr
from src.tokenizer import Tokenizer


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparâmetros. `epochs` é o horizonte do cosine schedule (não o número de épocas a executar)."""

    epochs: int
    embedding_dim: int
    n_heads: int
    n_blocks: int
    sequence_length: int
    d_ff: int
    batch_size: int
    learning_rate_max: float
    learning_rate_min: float
    weight_decay: float
    beta1: float
    beta2: float
    validation_batches: int
    val_fraction: float
    num_merges: int
    sample_prompt: str
    sample_tokens: int

    @classmethod
    def from_module(cls, cfg: ModuleType) -> "TrainConfig":
        return cls(
            epochs=cfg.EPOCHS,
            embedding_dim=cfg.EMBEDDING_DIM,
            n_heads=cfg.N_HEADS,
            n_blocks=cfg.N_BLOCKS,
            sequence_length=cfg.SEQUENCE_LENGTH,
            d_ff=cfg.D_FF,
            batch_size=cfg.BATCH_SIZE,
            learning_rate_max=cfg.LEARNING_RATE_MAX,
            learning_rate_min=cfg.LEARNING_RATE_MIN,
            weight_decay=cfg.WEIGHT_DECAY,
            beta1=cfg.BETA1,
            beta2=cfg.BETA2,
            validation_batches=cfg.VALIDATION_BATCHES,
            val_fraction=cfg.VAL_FRACTION,
            num_merges=cfg.NUM_MERGES,
            sample_prompt=cfg.SAMPLE_PROMPT,
            sample_tokens=cfg.SAMPLE_TOKENS,
        )


@dataclass(frozen=True)
class Paths:
    corpus_file: str
    tokenizer_file: str
    checkpoint_dir: str


@dataclass(frozen=True)
class RunOptions:
    epochs: int  # épocas a executar nesta chamada (retoma do último checkpoint)
    smoke: bool = False


def load_or_train_tokenizer(text: str, tokenizer_file: str, num_merges: int) -> Tokenizer:
    if os.path.exists(tokenizer_file):
        return Tokenizer.load(tokenizer_file)
    tok = Tokenizer()
    tok.train(text, num_merges)
    os.makedirs(os.path.dirname(tokenizer_file) or ".", exist_ok=True)
    tok.save(tokenizer_file)
    return tok


def latest_checkpoint(directory: str) -> str | None:
    files = sorted(glob.glob(os.path.join(directory, "epoch_*.npz")))
    return files[-1] if files else None


def validation_loss(
    model: GPT,
    val_ids: np.ndarray,
    rng: np.random.Generator,
    batches: int,
    batch_size: int,
    sequence_length: int,
) -> float:
    losses = []
    for _ in range(batches):
        x, t = get_batch(val_ids, batch_size, sequence_length, rng)
        _, loss = model.forward(x, t)
        losses.append(loss)
    return float(np.mean(losses))


def train(
    settings: TrainConfig,
    paths: Paths,
    options: RunOptions,
    on_save: Callable[[str], None] | None = None,
) -> str | None:
    """Treina até `options.epochs` e retorna o caminho do último checkpoint salvo.

    `on_save` é chamado após cada checkpoint (ex.: para dar commit em um Volume remoto).
    """
    text = open(paths.corpus_file, encoding="utf-8").read()
    tok = load_or_train_tokenizer(text, paths.tokenizer_file, settings.num_merges)
    ids = np.array(tok.encode(text), dtype=np.int32)
    split = int(len(ids) * (1 - settings.val_fraction))
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"Tokens: {len(ids):,} (treino {len(train_ids):,}, validação {len(val_ids):,}), vocab {tok.vocab_size}")

    model_kwargs = dict(
        vocab_size=tok.vocab_size,
        d_model=settings.embedding_dim,
        n_heads=settings.n_heads,
        n_blocks=settings.n_blocks,
        context_length=settings.sequence_length,
        d_ff=settings.d_ff,
    )
    os.makedirs(paths.checkpoint_dir, exist_ok=True)
    start_epoch = 0
    ckpt = latest_checkpoint(paths.checkpoint_dir)
    if ckpt is not None:
        model = GPT.load(ckpt, **model_kwargs)
        start_epoch = int(re.search(r"epoch_(\d+)", ckpt).group(1)) + 1
        print(f"Retomando de {ckpt} (época {start_epoch})")
    else:
        model = GPT(**model_kwargs)

    opt = AdamW(model.params_and_grads, settings.weight_decay, settings.beta1, settings.beta2, eps=1e-8)
    steps_per_epoch = len(train_ids) // (settings.batch_size * settings.sequence_length)
    total_steps = settings.epochs * steps_per_epoch
    val_batches = settings.validation_batches
    epoch_steps = steps_per_epoch
    if options.smoke:
        epoch_steps, val_batches = 5, 1

    rng = np.random.default_rng(0)
    val_rng = np.random.default_rng(123)
    step = start_epoch * steps_per_epoch
    last_ckpt: str | None = ckpt

    for epoch in tqdm(range(start_epoch, options.epochs), desc="épocas"):
        started = time.time()
        losses = []
        for _ in tqdm(range(epoch_steps), desc=f"época {epoch}", leave=False):
            lr = cosine_lr(step, total_steps, settings.learning_rate_max, settings.learning_rate_min)
            x, t = get_batch(train_ids, settings.batch_size, settings.sequence_length, rng)
            _, loss = model.forward(x, t)
            if loss > 10.0 or np.isnan(loss):
                print(f"AVISO: loss {loss} no passo {step}")
            model.backward()
            opt.step(lr, model.params_and_grads())
            model.zero_grad()
            losses.append(loss)
            step += 1

        train_loss = float(np.mean(losses))
        val_loss = validation_loss(
            model, val_ids, val_rng, val_batches, settings.batch_size, settings.sequence_length
        )
        elapsed = time.time() - started
        tokens_per_sec = epoch_steps * settings.batch_size * settings.sequence_length / elapsed
        print(
            f"época {epoch} | treino {train_loss:.4f} | validação {val_loss:.4f} "
            f"| lr {lr:.6f} | {tokens_per_sec:,.0f} tokens/s"
        )
        sample = generate(model, tok, settings.sample_prompt, settings.sample_tokens, temperature=0.8)
        print(f"amostra: {sample}")
        last_ckpt = os.path.join(paths.checkpoint_dir, f"epoch_{epoch:02d}.npz")
        model.save(last_ckpt)
        if on_save is not None:
            on_save(last_ckpt)

    print("Treino concluído.")
    return last_ckpt
