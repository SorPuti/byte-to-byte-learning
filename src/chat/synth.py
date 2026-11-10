"""Gera exemplos sintéticos de uso de ferramenta com rótulo verificável.

Para cada exemplo: escolhe uma ferramenta alvo, sorteia argumentos válidos pelo schema,
escreve a frase do usuário (de `examples_pt` da ferramenta, ou um modelo genérico) e
guarda a chamada correta. Também gera exemplos negativos, em que nenhuma ferramenta
deve ser chamada. O rótulo é conhecido por construção, então a avaliação é exata.
"""

import random
from typing import Any

from src.chat.format import render_reply, ToolCall
from src.chat.tools import ToolSpec, validate_arguments

_PT_NAMES = ["Maria", "João", "Ana Clara", "Pedro", "Beatriz", "Lucas", "Helena", "Rafael"]
_PT_CITIES = ["São Paulo", "Belo Horizonte", "Curitiba", "Recife", "Porto Alegre", "Salvador", "Manaus"]
_PT_WORDS = ["relatório", "contrato", "pedido", "reunião", "projeto", "pagamento", "cliente", "estoque"]

NEGATIVE_PT = [
    "Oi, tudo bem?",
    "Obrigado pela ajuda de ontem.",
    "Qual é a diferença entre 'mas' e 'mais'?",
    "Você pode me explicar o que é um substantivo?",
    "Conte uma curiosidade sobre o Brasil.",
]
NEGATIVE_REPLIES_PT = [
    "Tudo bem, obrigado! Em que posso ajudar?",
    "De nada! Se precisar de mais alguma coisa, é só falar.",
    "'Mas' indica oposição; 'mais' indica quantidade ou acréscimo.",
    "Substantivo é a palavra que nomeia seres, objetos, lugares ou ideias.",
    "O Brasil tem o maior território da América do Sul e fala português como idioma oficial.",
]


def _sample_string(name: str, rng: random.Random) -> str:
    lname = name.lower()
    if "data" in lname or "date" in lname:
        return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if "cidade" in lname or "city" in lname or "local" in lname:
        return rng.choice(_PT_CITIES)
    if "nome" in lname or "name" in lname or "participante" in lname or "pessoa" in lname:
        return rng.choice(_PT_NAMES)
    return rng.choice(_PT_WORDS)


def sample_value(schema: dict, rng: random.Random, name: str = "valor") -> Any:
    if "enum" in schema:
        return rng.choice(schema["enum"])
    kind = schema.get("type")
    if kind == "string":
        return _sample_string(name, rng)
    if kind == "integer":
        return rng.randint(1, 100)
    if kind == "number":
        return round(rng.uniform(0.5, 500.0), 2)
    if kind == "boolean":
        return rng.random() < 0.5
    if kind == "array":
        size = rng.randint(1, 3)
        return [sample_value(schema.get("items", {"type": "string"}), rng, name) for _ in range(size)]
    if kind == "object":
        return {k: sample_value(v, rng, k) for k, v in schema.get("properties", {}).items()}
    return None


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_render_value(v) for v in value)
    if isinstance(value, bool):
        return "sim" if value else "não"
    return str(value)


def build_user_text(tool: ToolSpec, arguments: dict, rng: random.Random) -> str:
    values = {k: _render_value(v) for k, v in arguments.items()}
    if tool.examples_pt:
        template = rng.choice(tool.examples_pt)
        try:
            return template.format(**values)
        except KeyError:
            pass  # template com placeholder que o exemplo não sorteou: cai no genérico
    pairs = ", ".join(f"{k} {v}" for k, v in values.items())
    return f"Preciso usar a ferramenta {tool.name} com {pairs}." if pairs else f"Preciso usar a ferramenta {tool.name}."


def _tools_subset(tools: list[ToolSpec], target: ToolSpec, rng: random.Random) -> list[ToolSpec]:
    others = [t for t in tools if t.name != target.name]
    k = rng.randint(0, min(4, len(others)))
    chosen = rng.sample(others, k)
    chosen.append(target)
    rng.shuffle(chosen)
    return chosen


def make_call_example(tools: list[ToolSpec], target: ToolSpec, rng: random.Random) -> dict:
    arguments = {}
    for key, schema in target.parameters.get("properties", {}).items():
        required = key in target.parameters.get("required", [])
        if required or rng.random() < 0.6:
            arguments[key] = sample_value(schema, rng, key)
    text = build_user_text(target, arguments, rng)
    calls = [ToolCall(name=target.name, arguments=arguments)]
    return {
        "tools": [t.to_dict() for t in _tools_subset(tools, target, rng)],
        "messages": [{"role": "user", "content": text}],
        "reply": {"text": "", "calls": [{"name": c.name, "arguments": c.arguments} for c in calls]},
    }


def make_negative_example(tools: list[ToolSpec], rng: random.Random) -> dict:
    i = rng.randrange(len(NEGATIVE_PT))
    chosen = rng.sample(tools, min(3, len(tools))) if tools else []
    return {
        "tools": [t.to_dict() for t in chosen],
        "messages": [{"role": "user", "content": NEGATIVE_PT[i]}],
        "reply": {"text": NEGATIVE_REPLIES_PT[i], "calls": []},
    }


def generate(tools: list[ToolSpec], n: int, negative_ratio: float = 0.2, seed: int = 0) -> list[dict]:
    if not tools:
        raise ValueError("é preciso pelo menos uma ferramenta")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < negative_ratio:
            out.append(make_negative_example(tools, rng))
        else:
            out.append(make_call_example(tools, rng.choice(tools), rng))
    return out


def example_is_valid(example: dict, tools: list[ToolSpec]) -> list[str]:
    """Confere que a chamada rotulada respeita o schema. Usado nos testes e para filtrar."""
    errors = []
    by_name = {t.name: t for t in tools}
    for call in example["reply"]["calls"]:
        spec = by_name.get(call["name"])
        if spec is None:
            errors.append(f"ferramenta fora do conjunto: {call['name']}")
            continue
        errors.extend(validate_arguments(spec.parameters, call["arguments"]))
    return errors


def render_target(example: dict) -> str:
    calls = [ToolCall(c["name"], c["arguments"]) for c in example["reply"]["calls"]]
    return render_reply(example["reply"]["text"], calls)
