import argparse

import numpy as np

from src.gpt import GPT, _softmax, model_kwargs
from src.tokenizer import Tokenizer


def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
) -> str:
    context_length = model.pos_emb.W.shape[0]
    ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        context = np.array([ids[-context_length:]])
        logits, _ = model.forward(context)
        probs = _softmax(logits[0, -1] / temperature)
        ids.append(int(np.random.choice(len(probs), p=probs)))
    return tokenizer.decode(ids)


def main() -> None:
    import config  # import local: a CLI usa os defaults de config.py; o núcleo não depende dele

    parser = argparse.ArgumentParser(description="Gera texto com um checkpoint do byte-to-byte.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default=config.SAMPLE_PROMPT)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()

    tok = Tokenizer.load(config.TOKENIZER_FILE)
    model = GPT.load(args.checkpoint, **model_kwargs(tok.vocab_size))
    print(generate(model, tok, args.prompt, args.tokens, args.temp))


if __name__ == "__main__":
    main()
