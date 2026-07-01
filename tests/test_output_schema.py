# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the inline outputs schema compiler."""

import pytest
from pydantic import ValidationError

from seclab_taskflow_agent.output_schema import (
    OutputSchemaError,
    build_output_model,
    validate_output,
)


class TestScalars:
    def test_scalar_fields(self):
        out = validate_output({"name": "str", "count": "int"}, {"name": "f", "count": 3})
        assert out == {"name": "f", "count": 3}

    def test_scalar_coercion(self):
        # pydantic coerces a numeric string to int
        out = validate_output({"count": "int"}, {"count": "42"})
        assert out["count"] == 42

    def test_synonyms(self):
        out = validate_output(
            {"a": "string", "b": "integer", "c": "number", "d": "boolean"},
            {"a": "x", "b": 1, "c": 1.5, "d": True},
        )
        assert out == {"a": "x", "b": 1, "c": 1.5, "d": True}

    def test_any_type(self):
        out = validate_output({"v": "any"}, {"v": {"nested": [1, 2]}})
        assert out["v"] == {"nested": [1, 2]}

    def test_optional_field_absent(self):
        out = validate_output({"name": "str", "note": "str?"}, {"name": "f"})
        assert out == {"name": "f", "note": None}

    def test_required_field_missing_raises(self):
        with pytest.raises(ValidationError):
            validate_output({"name": "str"}, {})

    def test_unsupported_scalar_raises(self):
        with pytest.raises(OutputSchemaError, match="unsupported type"):
            build_output_model("m", {"x": "decimal"})


class TestLists:
    def test_list_of_scalars(self):
        out = validate_output({"tags": "list[str]"}, {"tags": ["a", "b"]})
        assert out["tags"] == ["a", "b"]

    def test_bare_list_is_any(self):
        out = validate_output({"items": "list"}, {"items": [1, "a", {"k": 1}]})
        assert out["items"] == [1, "a", {"k": 1}]

    def test_list_type_with_scalar_items(self):
        out = validate_output({"nums": {"type": "list", "items": "int"}}, {"nums": [1, 2]})
        assert out["nums"] == [1, 2]

    def test_list_of_objects(self):
        schema = {"functions": {"type": "list", "items": {"name": "str", "body": "str"}}}
        value = {"functions": [{"name": "f", "body": "..."}, {"name": "g", "body": "!"}]}
        out = validate_output(schema, value)
        assert out["functions"][0] == {"name": "f", "body": "..."}
        assert out["functions"][1]["name"] == "g"

    def test_list_element_validation_error(self):
        with pytest.raises(ValidationError):
            validate_output({"nums": "list[int]"}, {"nums": ["not-an-int"]})


class TestNestedObjects:
    def test_nested_object_shorthand(self):
        schema = {"meta": {"author": "str", "version": "int"}}
        out = validate_output(schema, {"meta": {"author": "a", "version": 2}})
        assert out["meta"] == {"author": "a", "version": 2}

    def test_explicit_object_form(self):
        schema = {"meta": {"type": "object", "fields": {"author": "str"}}}
        out = validate_output(schema, {"meta": {"author": "a"}})
        assert out["meta"] == {"author": "a"}

    def test_object_missing_fields_raises(self):
        with pytest.raises(OutputSchemaError, match="requires a 'fields' mapping"):
            build_output_model("m", {"meta": {"type": "object"}})


class TestSchemaErrors:
    def test_empty_schema_raises(self):
        with pytest.raises(OutputSchemaError, match="non-empty mapping"):
            build_output_model("m", {})

    def test_invalid_field_name_raises(self):
        with pytest.raises(OutputSchemaError, match="invalid field name"):
            build_output_model("m", {"has-hyphen": "str"})

    def test_invalid_spec_type_raises(self):
        with pytest.raises(OutputSchemaError, match="invalid type spec"):
            build_output_model("m", {"x": 123})

    def test_extra_fields_preserved(self):
        # extra="allow": unknown fields in the value are kept
        out = validate_output({"name": "str"}, {"name": "f", "extra": 1})
        assert out["extra"] == 1
