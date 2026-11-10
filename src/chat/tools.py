"""Definição de ferramentas configuráveis e validação de argumentos.

Aceita o formato OpenAI ({"type": "function", "function": {...}}) ou o formato plano
({"name", "description", "parameters"}). `parameters` é um JSON Schema com `type: object`.
Validação suporta o subconjunto usado em function calling: type, properties, required,
enum, items e additionalProperties.
"""

from dataclasses import dataclass
from typing import Any

_NUMBER_TYPES = {"number", "integer"}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict
    examples_pt: tuple[str, ...] = ()  # frases de usuário com {placeholders}; opcional

    @classmethod
    def from_dict(cls, d: dict) -> "ToolSpec":
        f = d.get("function", d)
        params = f.get("parameters") or {"type": "object", "properties": {}}
        examples = d.get("examples_pt", f.get("examples_pt", ()))
        if not f.get("name"):
            raise ValueError("ferramenta sem 'name'")
        return cls(
            name=f["name"],
            description=f.get("description", ""),
            parameters=params,
            examples_pt=tuple(examples),
        )

    def to_dict(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def validate_arguments(schema: dict, value: Any, path: str = "args") -> list[str]:
    """Retorna a lista de erros (vazia se válido)."""
    errors: list[str] = []
    expected = schema.get("type")

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} fora de {schema['enum']}")
        return errors

    if expected == "object":
        if not isinstance(value, dict):
            return [f"{path}: esperado objeto, veio {type(value).__name__}"]
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: campo obrigatório ausente '{req}'")
        for key, sub in value.items():
            if key in props:
                errors.extend(validate_arguments(props[key], sub, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: campo desconhecido '{key}'")
        return errors

    if expected == "array":
        if not isinstance(value, list):
            return [f"{path}: esperado lista, veio {type(value).__name__}"]
        if "items" in schema:
            for i, item in enumerate(value):
                errors.extend(validate_arguments(schema["items"], item, f"{path}[{i}]"))
        return errors

    if expected == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: esperado string, veio {type(value).__name__}")
    elif expected == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: esperado boolean, veio {type(value).__name__}")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{path}: esperado inteiro, veio {type(value).__name__}")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path}: esperado número, veio {type(value).__name__}")
    elif expected == "null":
        if value is not None:
            errors.append(f"{path}: esperado null")
    return errors


def validate_call(tools: list[ToolSpec], name: str, arguments: Any) -> list[str]:
    by_name = {t.name: t for t in tools}
    if name not in by_name:
        return [f"ferramenta desconhecida '{name}'"]
    return validate_arguments(by_name[name].parameters, arguments)
