"""Avaliação de respostas do modelo contra exemplos com rótulo conhecido."""

from dataclasses import dataclass

from src.chat.format import parse_reply


@dataclass
class Score:
    parsed: bool        # saída tem formato válido (sem erro de JSON ou de marcador)
    name_ok: bool       # mesmo conjunto de ferramentas chamadas, na mesma ordem
    args_ok: bool       # argumentos iguais aos esperados (chamadas pareadas)
    no_call_ok: bool    # decidiu chamar ferramenta exatamente quando devia
    exact: bool         # tudo acima ao mesmo tempo


def score(example: dict, prediction: str) -> Score:
    gold = example["reply"]["calls"]
    parsed = parse_reply(prediction)
    got = [{"name": c.name, "arguments": c.arguments} for c in parsed.calls]

    fmt_ok = not parsed.errors
    name_ok = [c["name"] for c in got] == [c["name"] for c in gold]
    args_ok = name_ok and all(g["arguments"] == p["arguments"] for g, p in zip(gold, got))
    no_call_ok = (len(gold) == 0) == (len(got) == 0)
    exact = fmt_ok and name_ok and args_ok and no_call_ok
    return Score(parsed=fmt_ok, name_ok=name_ok, args_ok=args_ok, no_call_ok=no_call_ok, exact=exact)


def aggregate(scores: list[Score]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    rate = lambda attr: sum(getattr(s, attr) for s in scores) / n  # noqa: E731
    return {
        "n": n,
        "parse_rate": rate("parsed"),
        "name_acc": rate("name_ok"),
        "args_acc": rate("args_ok"),
        "no_call_acc": rate("no_call_ok"),
        "exact_match": rate("exact"),
    }
