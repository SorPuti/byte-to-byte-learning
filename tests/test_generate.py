import numpy as np
from src.generate import generate
from src.gpt import GPT
from src.tokenizer import Tokenizer

CFG = dict(vocab_size=50, d_model=16, n_heads=2, n_blocks=2, context_length=8, d_ff=32)


def _tok_and_model():
    tok = Tokenizer()
    tok.train("abcde " * 500, num_merges=20)
    model = GPT(**{**CFG, "vocab_size": tok.vocab_size})
    return tok, model


def test_generate_longer_than_prompt():
    tok, model = _tok_and_model()
    out = generate(model, tok, "abc", max_new_tokens=10, temperature=1.0)
    assert len(out) > len("abc")


def test_generate_returns_string():
    tok = Tokenizer()
    tok.train("hello world " * 500, num_merges=20)
    model = GPT(**{**CFG, "vocab_size": tok.vocab_size})
    assert isinstance(generate(model, tok, "hello", max_new_tokens=5, temperature=1.0), str)


def test_generate_near_zero_temperature_is_independent_of_seed():
    tok, model = _tok_and_model()
    np.random.seed(1)
    o1 = generate(model, tok, "abc", max_new_tokens=10, temperature=1e-6)
    np.random.seed(2)
    o2 = generate(model, tok, "abc", max_new_tokens=10, temperature=1e-6)
    assert o1 == o2


def test_generate_handles_prompt_longer_than_context():
    tok, model = _tok_and_model()
    out = generate(model, tok, "abcdeabcdeabcdeabcde", max_new_tokens=3, temperature=1.0)
    assert isinstance(out, str)
