from src.training import Paths, RunOptions, TrainConfig, latest_checkpoint, train

CORPUS = "Era uma vez um gato que dormia ao sol. " * 200

SMALL = TrainConfig(
    epochs=2,
    embedding_dim=16,
    n_heads=2,
    n_blocks=1,
    sequence_length=16,
    d_ff=32,
    batch_size=4,
    learning_rate_max=1e-3,
    learning_rate_min=1e-4,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    validation_batches=1,
    val_fraction=0.1,
    num_merges=20,
    sample_prompt="Era",
    sample_tokens=5,
)


def _paths(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    return Paths(str(corpus), str(tmp_path / "tok.json"), str(tmp_path / "ckpt"))


def test_smoke_run_writes_checkpoint_and_tokenizer(tmp_path):
    paths = _paths(tmp_path)
    saved = []
    last = train(SMALL, paths, RunOptions(epochs=1, smoke=True), on_save=saved.append)
    assert last == str(tmp_path / "ckpt" / "epoch_00.npz")
    assert saved == [last]
    assert (tmp_path / "tok.json").exists()


def test_second_run_resumes_from_latest_checkpoint(tmp_path):
    paths = _paths(tmp_path)
    train(SMALL, paths, RunOptions(epochs=1, smoke=True))
    last = train(SMALL, paths, RunOptions(epochs=2, smoke=True))
    assert last == str(tmp_path / "ckpt" / "epoch_01.npz")
    assert latest_checkpoint(paths.checkpoint_dir) == last


def test_from_module_reads_config_values():
    import config

    s = TrainConfig.from_module(config)
    assert s.epochs == config.EPOCHS
    assert s.embedding_dim == config.EMBEDDING_DIM
    assert s.d_ff == config.D_FF
    assert s.learning_rate_max == config.LEARNING_RATE_MAX
    assert s.sample_prompt == config.SAMPLE_PROMPT
