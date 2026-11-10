from src.chat import synth
from src.chat.evaluate import aggregate, score
from src.chat.format import render_reply
from src.chat.tools import ToolSpec

TOOLS = [
    ToolSpec.from_dict({
        "name": "agendar",
        "description": "Agenda reunião",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "string"},
                "duracao_min": {"type": "integer"},
                "participantes": {"type": "array", "items": {"type": "string"}},
                "prioridade": {"type": "string", "enum": ["baixa", "alta"]},
            },
            "required": ["data"],
            "additionalProperties": False,
        },
        "examples_pt": ["Marca uma reunião no dia {data} com {participantes}."],
    }),
    ToolSpec.from_dict({
        "name": "clima",
        "description": "Clima",
        "parameters": {"type": "object", "properties": {"cidade": {"type": "string"}}, "required": ["cidade"]},
    }),
]


def test_generated_calls_match_their_schemas():
    examples = synth.generate(TOOLS, 200, seed=3)
    assert len(examples) == 200
    for ex in examples:
        assert synth.example_is_valid(ex, TOOLS) == []


def test_generation_is_deterministic_for_same_seed():
    assert synth.generate(TOOLS, 20, seed=7) == synth.generate(TOOLS, 20, seed=7)


def test_negative_examples_have_no_calls_and_exist():
    examples = synth.generate(TOOLS, 300, negative_ratio=0.5, seed=1)
    negatives = [e for e in examples if not e["reply"]["calls"]]
    assert negatives and all(e["reply"]["text"] for e in negatives)


def test_user_text_uses_template_values():
    ex = synth.generate(TOOLS[:1], 50, negative_ratio=0.0, seed=2)[0]
    args = ex["reply"]["calls"][0]["arguments"]
    assert args["data"] in ex["messages"][0]["content"]


def test_perfect_prediction_scores_exact():
    ex = synth.generate(TOOLS, 1, negative_ratio=0.0, seed=4)[0]
    assert score(ex, synth.render_target(ex)).exact


def test_wrong_argument_is_partial_credit():
    ex = synth.generate(TOOLS, 1, negative_ratio=0.0, seed=4)[0]
    call = ex["reply"]["calls"][0]
    wrong = dict(call["arguments"], data="1999-01-01")
    pred = render_reply("", [])  # sem chamada
    s = score(ex, pred)
    assert not s.exact and not s.no_call_ok
    from src.chat.format import ToolCall

    s2 = score(ex, render_reply("", [ToolCall(call["name"], wrong)]))
    assert s2.parsed and s2.name_ok and not s2.args_ok and not s2.exact


def test_hallucinated_call_on_negative_example_is_caught():
    neg = next(e for e in synth.generate(TOOLS, 100, negative_ratio=1.0, seed=5))
    from src.chat.format import ToolCall

    s = score(neg, render_reply("", [ToolCall("clima", {"cidade": "Recife"})]))
    assert not s.no_call_ok and not s.exact


def test_aggregate_rates():
    ex = synth.generate(TOOLS, 10, negative_ratio=0.0, seed=6)
    agg = aggregate([score(e, synth.render_target(e)) for e in ex])
    assert agg == {"n": 10, "parse_rate": 1.0, "name_acc": 1.0, "args_acc": 1.0, "no_call_acc": 1.0, "exact_match": 1.0}
    assert aggregate([]) == {"n": 0}
