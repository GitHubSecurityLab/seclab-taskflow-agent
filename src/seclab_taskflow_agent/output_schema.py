# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Compile inline ``outputs`` schemas into Pydantic models.

A task can declare a typed output contract inline:

.. code-block:: yaml

    outputs:
      functions:
        type: list
        items:
          name: str
          body: str

This module turns such a schema into a generated Pydantic model so a task's
produced value can be validated and coerced before it is passed to downstream
tasks as ``outputs.<id>``.

Supported field type specs:

* Scalars (as strings): ``str``, ``int``, ``float``, ``bool``, ``any`` (plus
  the synonyms ``string``/``integer``/``number``/``boolean``). A trailing
  ``?`` marks the field optional, e.g. ``str?``.
* Lists (as strings): ``list`` (list of any) or ``list[T]`` where ``T`` is a
  scalar type name, e.g. ``list[str]``.
* Nested objects (as a mapping): a mapping whose keys are sub-field specs, or
  the explicit ``{type: object, fields: {...}}`` form.
* Lists of objects (as a mapping): ``{type: list, items: <spec>}`` where
  ``items`` is any spec (a scalar string, a nested object mapping, etc.).
"""

from __future__ import annotations

__all__ = ["OutputSchemaError", "build_output_model", "validate_output"]

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, create_model

_SCALARS: dict[str, Any] = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "any": Any,
}


class OutputSchemaError(ValueError):
    """Raised when an ``outputs`` schema is malformed or unsupported."""


def _scalar_type(name: str, field: str) -> Any:
    key = name.strip().lower()
    if key not in _SCALARS:
        raise OutputSchemaError(f"unsupported type {name!r} in field {field!r}")
    return _SCALARS[key]


def _spec_to_type(field: str, spec: Any, model_name: str) -> tuple[Any, bool]:
    """Resolve a field spec into ``(python_type, optional)``."""
    if isinstance(spec, str):
        s = spec.strip()
        optional = s.endswith("?")
        if optional:
            s = s[:-1].strip()
        low = s.lower()
        if low == "list":
            return list, optional
        if low.startswith("list[") and low.endswith("]"):
            inner = s[5:-1].strip()
            return list[_scalar_type(inner, field)], optional
        return _scalar_type(s, field), optional

    if isinstance(spec, dict):
        optional = bool(spec.get("optional", False))
        declared = spec.get("type")
        if declared is None:
            # Nested object shorthand: the mapping's keys are sub-fields.
            return _build_model(f"{model_name}_{field}", spec), optional
        declared = str(declared).lower()
        if declared == "list":
            items = spec.get("items", "any")
            inner_type, _ = _spec_to_type(field, items, f"{model_name}_{field}")
            return list[inner_type], optional
        if declared == "object":
            fields = spec.get("fields")
            if not isinstance(fields, dict):
                raise OutputSchemaError(
                    f"object field {field!r} requires a 'fields' mapping"
                )
            return _build_model(f"{model_name}_{field}", fields), optional
        # Scalar declared via the ``type`` key, e.g. {type: str}.
        return _scalar_type(declared, field), optional

    raise OutputSchemaError(f"invalid type spec for field {field!r}: {spec!r}")


def _build_model(name: str, fields: Any) -> type[BaseModel]:
    if not isinstance(fields, dict) or not fields:
        raise OutputSchemaError(f"schema {name!r} must be a non-empty mapping of fields")
    field_defs: dict[str, Any] = {}
    for fname, fspec in fields.items():
        if not isinstance(fname, str) or not fname.isidentifier():
            raise OutputSchemaError(f"invalid field name {fname!r} in schema {name!r}")
        py_type, optional = _spec_to_type(fname, fspec, name)
        if optional:
            field_defs[fname] = (Optional[py_type], None)
        else:
            field_defs[fname] = (py_type, ...)
    return create_model(name, __config__=ConfigDict(extra="allow"), **field_defs)


def build_output_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Compile an ``outputs`` schema mapping into a Pydantic model class.

    Raises:
        OutputSchemaError: If the schema is malformed or uses an unsupported
            type. Callers can use this to validate schemas at load time.
    """
    return _build_model(name, schema)


def validate_output(
    schema: dict[str, Any], value: Any, model_name: str = "TaskOutput"
) -> dict[str, Any]:
    """Validate/coerce *value* against *schema*, returning a plain dict.

    Raises:
        OutputSchemaError: If the schema itself is invalid.
        pydantic.ValidationError: If *value* does not satisfy the schema.
    """
    model = build_output_model(model_name, schema)
    obj = model.model_validate(value)
    return obj.model_dump()
