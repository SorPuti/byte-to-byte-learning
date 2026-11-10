import pytest

from src.chat.tools import ToolSpec, validate_arguments, validate_call

SCHEMA = {
    "type": "object",
    "properties": {
        "cidade": {"type": "string"},
        "dias": {"type": "integer"},
        "unidade": {"type": "string", "enum": ["C", "F"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "ativo": {"type": "boolean"},
    },
    "required": ["cidade"],
    "additionalProperties": False,
}


def test_from_dict_accepts_openai_and_flat_formats():
    openai = ToolSpec.from_dict({"type": "function", "function": {"name": "clima", "description": "d", "parameters": SCHEMA}})
    flat = ToolSpec.from_dict({"name": "clima", "description": "d", "parameters": SCHEMA})
    assert openai == flat
    assert openai.to_dict()["function"]["name"] == "clima"


def test_from_dict_requires_name():
    with pytest.raises(ValueError):
        ToolSpec.from_dict({"description": "sem nome"})


def test_valid_arguments_pass():
    assert validate_arguments(SCHEMA, {"cidade": "Recife", "dias": 3, "unidade": "C", "tags": ["a"], "ativo": True}) == []


def test_missing_required_field_is_reported():
    errors = validate_arguments(SCHEMA, {"dias": 3})
    assert any("obrigatório ausente 'cidade'" in e for e in errors)


def test_wrong_types_are_reported():
    errors = validate_arguments(SCHEMA, {"cidade": 5, "dias": 2.5, "ativo": "sim"})
    assert len(errors) == 3


def test_bool_is_not_integer():
    assert validate_arguments(SCHEMA, {"cidade": "x", "dias": True}) != []


def test_enum_violation():
    assert validate_arguments(SCHEMA, {"cidade": "x", "unidade": "K"}) != []


def test_unknown_field_rejected_when_additional_properties_false():
    assert any("desconhecido" in e for e in validate_arguments(SCHEMA, {"cidade": "x", "extra": 1}))


def test_array_items_are_checked():
    assert validate_arguments(SCHEMA, {"cidade": "x", "tags": ["ok", 3]}) != []


def test_validate_call_unknown_tool():
    tools = [ToolSpec("clima", "", SCHEMA)]
    assert validate_call(tools, "outra", {}) == ["ferramenta desconhecida 'outra'"]
    assert validate_call(tools, "clima", {"cidade": "Recife"}) == []
