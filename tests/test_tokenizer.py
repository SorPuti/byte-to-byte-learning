import pytest
from src.tokenizer import Tokenizer

SAMPLE = "Dom Casmurro nasceu numa tarde de verão. Dom Casmurro foi embora."


def test_vocab_grows_with_merges():
    tok = Tokenizer()
    tok.train(SAMPLE, num_merges=10)
    assert tok.vocab_size > len(set(SAMPLE))


def test_encode_decode_roundtrip():
    tok = Tokenizer()
    tok.train(SAMPLE, num_merges=10)
    assert tok.decode(tok.encode(SAMPLE)) == SAMPLE


def test_encode_returns_ints():
    tok = Tokenizer()
    tok.train(SAMPLE, num_merges=5)
    ids = tok.encode("Dom")
    assert all(isinstance(i, int) for i in ids)


def test_save_load_roundtrip(tmp_path):
    tok = Tokenizer()
    tok.train(SAMPLE, num_merges=10)
    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = Tokenizer.load(path)
    assert tok2.decode(tok2.encode(SAMPLE)) == SAMPLE
    assert tok2.vocab_size == tok.vocab_size


def test_stability_large_values():
    text = "".join(chr(i) for i in range(32, 127)) * 3
    tok = Tokenizer()
    tok.train(text, num_merges=20)
    assert tok.decode(tok.encode(text)) == text
