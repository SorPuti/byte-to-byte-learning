import json

from src.chat import format as f
from src.chat.format import Message, ToolCall
from src.chat.tools import ToolSpec

TOOL = ToolSpec("clima", "Consulta clima", {"type": "object", "properties": {"cidade": {"type": "string"}}, "required": ["cidade"]})


def test_render_reply_round_trips_through_parser():
    text = f.render_reply("Vou consultar.", [ToolCall("clima", {"cidade": "Recife"})])
    parsed = f.parse_reply(text)
    assert parsed.errors == []
    assert parsed.text == "Vou consultar."
    assert parsed.calls == [ToolCall("clima", {"cidade": "Recife"})]


def test_parse_cuts_at_end_token():
    parsed = f.parse_reply("Olá!" + f.END + "lixo depois do fim")
    assert parsed.text == "Olá!"
    assert parsed.calls == []


def test_parse_reports_invalid_json_without_raising():
    parsed = f.parse_reply(f.CALL_OPEN + "{nao é json" + f.CALL_CLOSE + f.END)
    assert parsed.calls == []
    assert any("JSON inválido" in e for e in parsed.errors)


def test_parse_reports_missing_name_and_bad_arguments():
    bad = f.CALL_OPEN + json.dumps({"arguments": {}}) + f.CALL_CLOSE
    parsed = f.parse_reply(bad)
    assert parsed.errors == ["chamada sem 'name' string"]
    bad_args = f.CALL_OPEN + json.dumps({"name": "x", "arguments": [1]}) + f.CALL_CLOSE
    assert f.parse_reply(bad_args).errors == ["'arguments' não é objeto"]


def test_unclosed_call_marker_is_an_error():
    assert "marcador de chamada sem par" in f.parse_reply(f.CALL_OPEN + '{"name":"x"}').errors


def test_render_conversation_includes_tools_and_ends_on_assistant_marker():
    prompt = f.render_conversation([Message("user", "Oi")], tools=[TOOL], system="Seja breve.")
    assert prompt.startswith(f.SYSTEM + "Seja breve." + f.TOOLS_OPEN)
    assert '"name": "clima"' in prompt
    assert prompt.endswith(f.USER + "Oi" + f.END + f.ASSISTANT)


def test_render_conversation_without_tools_has_no_system_block():
    prompt = f.render_conversation([Message("user", "Oi")])
    assert f.SYSTEM not in prompt
    assert prompt == f.USER + "Oi" + f.END + f.ASSISTANT


def test_tool_result_role_is_rendered():
    prompt = f.render_conversation([Message("tool", '{"temp": 30}')])
    assert prompt.startswith(f.TOOL + '{"temp": 30}' + f.END)


def test_unknown_role_raises():
    import pytest

    with pytest.raises(ValueError):
        f.render_conversation([Message("robô", "x")])


def test_special_tokens_are_unique():
    assert len(set(f.SPECIAL_TOKENS)) == len(f.SPECIAL_TOKENS)
