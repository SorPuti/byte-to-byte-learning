"""Formato de conversa com marcadores de papel e de chamada de ferramenta.

Os marcadores são strings literais. Ao treinar, cada um vira um token especial do tokenizador.
"""

import json
import re
from dataclasses import dataclass, field

from src.chat.tools import ToolSpec

SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
TOOL = "<|tool|>"
END = "<|end|>"
TOOLS_OPEN = "<|tools|>"
TOOLS_CLOSE = "<|end_tools|>"
CALL_OPEN = "<|call|>"
CALL_CLOSE = "<|end_call|>"

SPECIAL_TOKENS = [SYSTEM, USER, ASSISTANT, TOOL, END, TOOLS_OPEN, TOOLS_CLOSE, CALL_OPEN, CALL_CLOSE]


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str = ""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ParsedReply:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def render_tools_block(tools: list[ToolSpec]) -> str:
    payload = json.dumps([t.to_dict() for t in tools], ensure_ascii=False, sort_keys=True)
    return f"{TOOLS_OPEN}{payload}{TOOLS_CLOSE}"


def render_call(name: str, arguments: dict) -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True)
    return f"{CALL_OPEN}{payload}{CALL_CLOSE}"


def render_reply(text: str, calls: list[ToolCall]) -> str:
    """Texto da resposta do assistente (o que o modelo deve gerar), terminado por END."""
    return text + "".join(render_call(c.name, c.arguments) for c in calls) + END


def render_conversation(messages: list[Message], tools: list[ToolSpec] = (), system: str = "") -> str:
    """Prompt completo, terminando no marcador do assistente (ponto em que o modelo começa a gerar)."""
    parts = []
    if system or tools:
        block = system + (render_tools_block(list(tools)) if tools else "")
        parts.append(f"{SYSTEM}{block}{END}")
    for m in messages:
        if m.role == "user":
            parts.append(f"{USER}{m.content}{END}")
        elif m.role == "assistant":
            parts.append(f"{ASSISTANT}{m.content}{END}")
        elif m.role == "tool":
            parts.append(f"{TOOL}{m.content}{END}")
        else:
            raise ValueError(f"papel desconhecido: {m.role}")
    parts.append(ASSISTANT)
    return "".join(parts)


_CALL_RE = re.compile(re.escape(CALL_OPEN) + r"(.*?)" + re.escape(CALL_CLOSE), re.S)


def parse_reply(text: str) -> ParsedReply:
    """Interpreta a saída do modelo. Corta em END e extrai as chamadas.

    Erros de formato não levantam exceção: ficam em `errors`, para a avaliação contar.
    """
    if END in text:
        text = text[: text.index(END)]
    calls: list[ToolCall] = []
    errors: list[str] = []
    for raw in _CALL_RE.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"JSON inválido na chamada: {exc.msg}")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            errors.append("chamada sem 'name' string")
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            errors.append("'arguments' não é objeto")
            continue
        calls.append(ToolCall(name=obj["name"], arguments=args))
    prose = _CALL_RE.sub("", text).strip()
    if CALL_OPEN in prose or CALL_CLOSE in prose:
        errors.append("marcador de chamada sem par")
    return ParsedReply(text=prose, calls=calls, errors=errors)
