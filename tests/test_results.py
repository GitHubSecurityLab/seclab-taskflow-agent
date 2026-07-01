# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the neutral result representation and per-run store."""

import json

import pytest

from seclab_taskflow_agent.results import (
    ResultStore,
    ToolResult,
    decode_tool_result,
    normalize_openai_tool_output,
)


class TestNormalizeOpenAIToolOutput:
    """normalize_openai_tool_output handles every backend/serialisation shape."""

    def test_single_text_content_part(self):
        """openai-agents single text content: {"type":"text","text":...}."""
        raw = json.dumps({"type": "text", "text": "[1, 2, 3]", "annotations": None})
        tr = normalize_openai_tool_output(raw)
        assert tr.text == "[1, 2, 3]"
        assert tr.structured is None

    def test_lone_text_envelope(self):
        """Historical bridge envelope: {"text": ...}."""
        raw = json.dumps({"text": json.dumps({"a": 1})})
        tr = normalize_openai_tool_output(raw)
        assert tr.text == '{"a": 1}'
        assert tr.structured is None

    def test_structured_content_object(self):
        """A structured object (no text-part shape) becomes structured."""
        raw = json.dumps({"functions": [{"name": "f", "body": "..."}]})
        tr = normalize_openai_tool_output(raw)
        assert tr.text is None
        assert tr.structured == {"functions": [{"name": "f", "body": "..."}]}

    def test_multiple_text_parts_joined(self):
        """Multiple text content parts join into a single text payload."""
        raw = json.dumps([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
        tr = normalize_openai_tool_output(raw)
        assert tr.text == "a\nb"

    def test_mixed_content_list_is_structured(self):
        """A list whose items are not all text parts stays structured."""
        raw = json.dumps([{"type": "text", "text": "a"}, {"type": "image", "data": "x"}])
        tr = normalize_openai_tool_output(raw)
        assert tr.text is None
        assert isinstance(tr.structured, list)

    def test_empty_content_list(self):
        """openai-agents serialises empty content as "[]"."""
        tr = normalize_openai_tool_output("[]")
        assert tr.text is None
        assert tr.structured == []

    def test_non_json_text(self):
        """A non-JSON string is preserved verbatim as text."""
        tr = normalize_openai_tool_output("plain answer, not json")
        assert tr.text == "plain answer, not json"

    def test_already_a_tool_result_passthrough(self):
        tr_in = ToolResult(tool_name="x", text="hi")
        assert normalize_openai_tool_output(tr_in) is tr_in

    def test_non_string_structured_object(self):
        """A pre-parsed dict input is treated as structured content."""
        tr = normalize_openai_tool_output({"k": "v"}, tool_name="t")
        assert tr.structured == {"k": "v"}
        assert tr.tool_name == "t"

    def test_scalar_is_structured(self):
        tr = normalize_openai_tool_output("42")
        assert tr.structured == 42


class TestDecodeToolResult:
    """decode_tool_result is the single decode path for consuming a result."""

    def test_structured_preferred(self):
        tr = ToolResult(structured={"functions": [1, 2]})
        assert decode_tool_result(tr) == {"functions": [1, 2]}

    def test_json_text_parsed(self):
        tr = ToolResult(text=json.dumps([1, 2, 3]))
        assert decode_tool_result(tr) == [1, 2, 3]

    def test_non_json_text_raises(self):
        tr = ToolResult(text="not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            decode_tool_result(tr)

    def test_empty_result_raises(self):
        with pytest.raises(ValueError, match="neither text nor structured"):
            decode_tool_result(ToolResult())

    def test_roundtrip_from_openai_single_text(self):
        """The common path: single text content carrying a JSON array."""
        raw = json.dumps({"type": "text", "text": json.dumps([{"name": "a"}])})
        value = decode_tool_result(normalize_openai_tool_output(raw))
        assert value == [{"name": "a"}]


class TestResultStore:
    """ResultStore holds ordered results and named outputs with snapshotting."""

    def test_record_and_last(self):
        s = ResultStore()
        assert s.last() is None
        s.record(ToolResult(text="a"))
        s.record(ToolResult(text="b"))
        assert s.last().text == "b"

    def test_pop_last(self):
        s = ResultStore()
        s.record(ToolResult(text="a"))
        s.record(ToolResult(text="b"))
        s.pop_last()
        assert s.last().text == "a"
        s.pop_last()
        assert s.last() is None
        # popping an empty store is a no-op
        s.pop_last()
        assert s.last() is None

    def test_named_outputs(self):
        s = ResultStore()
        s.set_output("list_functions", {"functions": [1, 2]})
        assert s.outputs["list_functions"] == {"functions": [1, 2]}

    def test_snapshot_roundtrip(self):
        s = ResultStore()
        s.record(ToolResult(tool_name="t", text=json.dumps([1, 2])))
        s.record(ToolResult(structured={"k": "v"}))
        s.set_output("out", {"functions": ["a"]})

        snap = s.snapshot()
        # snapshot must be JSON-serialisable for session persistence
        json.dumps(snap)

        restored = ResultStore.from_snapshot(snap)
        assert restored.last().structured == {"k": "v"}
        assert restored.results[0].text == json.dumps([1, 2])
        assert restored.outputs["out"] == {"functions": ["a"]}

    def test_from_snapshot_none_is_empty(self):
        s = ResultStore.from_snapshot(None)
        assert s.last() is None
        assert s.outputs == {}
