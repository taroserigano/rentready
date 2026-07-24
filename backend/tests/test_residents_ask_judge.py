"""Tests for the Residents-chat LLM-as-judge harness.

Covers the content-coalescing regression found while running the judge live:
an extended-thinking response is a list of content blocks — a
``{"type": "thinking", ...}`` block with no "text" key, then the real
``{"type": "text", "text": "..."}`` block. A naive ``str(b)`` join stringifies
the whole thinking-block dict ahead of the JSON, breaking ``_first_json_obj``.
"""

from evals import residents_ask_judge as j


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        return _FakeResponse(self._content)


def test_invoke_coalesces_plain_string():
    llm = _FakeLLM('{"faithfulness": 5}')
    assert j._invoke(llm, "sys", "human") == '{"faithfulness": 5}'


def test_invoke_coalesces_text_only_blocks():
    llm = _FakeLLM([{"type": "text", "text": '{"faithfulness": 4}'}])
    assert j._invoke(llm, "sys", "human") == '{"faithfulness": 4}'


def test_invoke_skips_thinking_block_before_text_block():
    """The exact regression: a thinking block (no "text" key) precedes the
    real text block. The old naive str(b) join broke JSON parsing here."""
    llm = _FakeLLM([
        {"type": "thinking", "thinking": "reasoning...", "signature": "abc123"},
        {"type": "text", "text": '{"faithfulness": 3, "safe": true, "helpful": 5}'},
    ])
    raw = j._invoke(llm, "sys", "human")
    assert raw == '{"faithfulness": 3, "safe": true, "helpful": 5}'
    parsed = j._first_json_obj(raw)
    assert parsed == {"faithfulness": 3, "safe": True, "helpful": 5}


def test_first_json_obj_parses_coalesced_output():
    llm = _FakeLLM([
        {"type": "thinking", "thinking": "..."},
        {"type": "text", "text": '{"faithfulness": 5, "safe": true, "helpful": 5}'},
    ])
    obj = j._first_json_obj(j._invoke(llm, "sys", "human"))
    assert obj["faithfulness"] == 5
    assert obj["safe"] is True
