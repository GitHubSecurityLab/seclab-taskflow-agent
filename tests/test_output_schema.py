# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the inline outputs JSON Schema validation."""

import pytest
from jsonschema.exceptions import ValidationError

from seclab_taskflow_agent.output_schema import (
    OutputSchemaError,
    validate_output,
    validate_output_schema,
)


def _obj(props, required=None, **extra):
    s = {"type": "object", "properties": props, **extra}
    if required is not None:
        s["required"] = required
    return s


class TestValidateOutput:
    def test_valid_returns_value_unchanged(self):
        schema = _obj({"name": {"type": "string"}, "count": {"type": "integer"}}, ["name", "count"])
        value = {"name": "f", "count": 3}
        assert validate_output(schema, value) == value

    def test_strict_no_coercion(self):
        # JSON Schema validation does not coerce: a numeric string is not an integer.
        schema = _obj({"count": {"type": "integer"}}, ["count"])
        with pytest.raises(ValidationError):
            validate_output(schema, {"count": "42"})

    def test_missing_required_raises(self):
        schema = _obj({"name": {"type": "string"}}, ["name"])
        with pytest.raises(ValidationError):
            validate_output(schema, {})

    def test_array_of_scalars(self):
        schema = _obj({"tags": {"type": "array", "items": {"type": "string"}}}, ["tags"])
        assert validate_output(schema, {"tags": ["a", "b"]}) == {"tags": ["a", "b"]}

    def test_array_element_type_error(self):
        schema = _obj({"nums": {"type": "array", "items": {"type": "integer"}}}, ["nums"])
        with pytest.raises(ValidationError):
            validate_output(schema, {"nums": ["not-an-int"]})

    def test_array_of_objects(self):
        item = _obj({"name": {"type": "string"}, "body": {"type": "string"}}, ["name", "body"])
        schema = _obj({"functions": {"type": "array", "items": item}}, ["functions"])
        value = {"functions": [{"name": "f", "body": "..."}, {"name": "g", "body": "!"}]}
        assert validate_output(schema, value) == value

    def test_nested_object(self):
        meta = _obj({"author": {"type": "string"}, "version": {"type": "integer"}}, ["author", "version"])
        schema = _obj({"meta": meta}, ["meta"])
        value = {"meta": {"author": "a", "version": 2}}
        assert validate_output(schema, value) == value

    def test_top_level_array(self):
        # The bespoke DSL could not express a top-level (non-object) output.
        schema = {"type": "array", "items": {"type": "string"}}
        assert validate_output(schema, ["a", "b"]) == ["a", "b"]


class TestExpressiveness:
    """Capabilities the old type mini-DSL lacked."""

    def test_enum(self):
        schema = _obj({"sev": {"enum": ["low", "medium", "high"]}}, ["sev"])
        assert validate_output(schema, {"sev": "high"}) == {"sev": "high"}
        with pytest.raises(ValidationError):
            validate_output(schema, {"sev": "critical"})

    def test_numeric_constraints(self):
        schema = _obj({"score": {"type": "integer", "minimum": 0, "maximum": 10}}, ["score"])
        assert validate_output(schema, {"score": 7}) == {"score": 7}
        with pytest.raises(ValidationError):
            validate_output(schema, {"score": 99})

    def test_dict_dynamic_keys(self):
        # findings-by-file: an object with arbitrary keys, each a list of strings.
        schema = {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}}
        value = {"a.c": ["overflow"], "b.c": ["uaf"]}
        assert validate_output(schema, value) == value

    def test_additional_properties_false_is_strict(self):
        schema = _obj({"name": {"type": "string"}}, ["name"], additionalProperties=False)
        with pytest.raises(ValidationError):
            validate_output(schema, {"name": "f", "hallucinated": 1})

    def test_reserved_field_names_are_safe(self):
        # `type`/`items`/`fields` are ordinary property names in JSON Schema; the
        # old DSL misparsed (and dropped fields of) an object named like a keyword.
        schema = _obj({"type": {"type": "string"}, "value": {"type": "string"}}, ["type", "value"])
        value = {"type": "cwe", "value": "787"}
        assert validate_output(schema, value) == value


class TestSchemaWellFormedness:
    def test_valid_schema_accepted(self):
        validate_output_schema(_obj({"x": {"type": "string"}}))

    def test_non_dict_schema_raises(self):
        with pytest.raises(OutputSchemaError, match="non-empty"):
            validate_output_schema("nope")

    def test_empty_schema_raises(self):
        with pytest.raises(OutputSchemaError, match="non-empty"):
            validate_output_schema({})

    def test_malformed_schema_raises(self):
        with pytest.raises(OutputSchemaError, match="invalid 'outputs' schema"):
            validate_output_schema({"type": "not-a-json-schema-type"})
