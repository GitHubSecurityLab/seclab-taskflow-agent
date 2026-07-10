# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Validate task outputs against an inline JSON Schema.

A task can declare a typed output contract inline as a JSON Schema (Draft
2020-12), authored directly in YAML:

.. code-block:: yaml

    outputs:
      type: object
      properties:
        findings:
          type: array
          items:
            type: object
            properties:
              file: {type: string}
              severity: {type: string, enum: [low, medium, high]}
            required: [file, severity]
      required: [findings]

The task's produced value is validated against the schema before it is passed
to downstream tasks as ``outputs.<id>``. Using JSON Schema (validated with the
``jsonschema`` library) rather than a bespoke type language gives enums,
numeric/string constraints, unions, objects with dynamic keys, and ``$ref``
reuse for free, and the schema itself is checked for well-formedness when the
taskflow loads.
"""

from __future__ import annotations

__all__ = ["OutputSchemaError", "validate_output", "validate_output_schema"]

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


class OutputSchemaError(ValueError):
    """Raised when an ``outputs`` schema is not a well-formed JSON Schema."""


def validate_output_schema(schema: Any) -> None:
    """Check that *schema* is a well-formed JSON Schema, raising on failure.

    Called at taskflow load time so a malformed output contract fails fast,
    before any model calls are made.

    Raises:
        OutputSchemaError: If *schema* is not a non-empty, valid JSON Schema.
    """
    if not isinstance(schema, dict) or not schema:
        raise OutputSchemaError("'outputs' must be a non-empty JSON Schema object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise OutputSchemaError(f"invalid 'outputs' schema: {exc.message}") from exc


def validate_output(schema: dict[str, Any], value: Any) -> Any:
    """Validate *value* against the JSON Schema *schema* and return it unchanged.

    JSON Schema validation does not coerce, so a value whose types do not
    already match the contract is a failure. This is deliberately stricter than
    permissive coercion: it surfaces malformed model output instead of silently
    reshaping it.

    Raises:
        OutputSchemaError: If *schema* itself is malformed.
        jsonschema.ValidationError: If *value* does not satisfy *schema*.
    """
    validate_output_schema(schema)
    Draft202012Validator(schema).validate(value)
    return value
