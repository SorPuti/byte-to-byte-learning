import contextlib
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import pytest

import src.backends.local as local_module
from src.backends.local import LocalBackend
from src.backends.modal_backend import ModalBackend
from src.training import Paths, RunOptions, TrainConfig

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = TrainConfig.from_module(__import__("config"))


def test_local_backend_ensures_corpus_then_trains(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(local_module, "train", lambda s, p, o: ("treinou", p.checkpoint_dir))
    paths = Paths(str(tmp_path / "c.txt"), str(tmp_path / "t.json"), str(tmp_path / "ck"))
    backend = LocalBackend(paths, ensure_corpus=calls.append)
    result = backend.run(SETTINGS, RunOptions(epochs=1))
    assert calls == [paths.corpus_file]
    assert result == ("treinou", paths.checkpoint_dir)


def test_modal_backend_sends_plain_dicts(monkeypatch):
    received = {}

    class FakeApp:
        def run(self):
            return contextlib.nullcontext()

    class FakeFunction:
        def remote(self, settings, options):
            received["settings"] = settings
            received["options"] = options
            return "/checkpoints/epoch_00.npz"

    fake = types.ModuleType("modal_app")
    fake.app = FakeApp()
    fake.train_remote = FakeFunction()
    monkeypatch.setitem(sys.modules, "modal_app", fake)

    out = ModalBackend().run(SETTINGS, RunOptions(epochs=2, smoke=True))
    assert out == "/checkpoints/epoch_00.npz"
    assert received["settings"] == asdict(SETTINGS)
    assert received["options"] == {"epochs": 2, "smoke": True}


def test_build_backend_unknown_name_raises():
    import train

    with pytest.raises(ValueError):
        train.build_backend("gpu-magico", "ckpt")


def test_core_and_cli_do_not_import_modal():
    code = (
        "import sys, train, src.training, src.backends;"
        "assert 'modal' not in sys.modules and 'modal_app' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
