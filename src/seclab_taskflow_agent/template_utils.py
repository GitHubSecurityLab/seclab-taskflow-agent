# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Jinja2 template utilities for taskflow template rendering."""

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import jinja2

__all__ = ["PromptLoader", "create_jinja_environment", "env_function", "evaluate_expression", "render_template"]

if TYPE_CHECKING:
    from .available_tools import AvailableTools

from .available_tools import BadToolNameError


class PromptLoader(jinja2.BaseLoader):
    """Custom Jinja2 loader for reusable prompts."""

    def __init__(self, available_tools: "AvailableTools") -> None:
        """Initialize the prompt loader.

        Args:
            available_tools: AvailableTools instance for prompt loading
        """
        self.available_tools = available_tools

    def get_source(
        self, environment: jinja2.Environment, template: str
    ) -> tuple[str, str | None, Callable[[], bool]]:
        """Load prompt from available_tools by path.

        Args:
            environment: Jinja2 environment
            template: Template path (e.g., 'examples.prompts.example_prompt')

        Returns:
            Tuple of (source, filename, uptodate_func)

        Raises:
            jinja2.TemplateNotFound: If prompt not found
        """
        del environment # unused arg
        try:
            prompt_data = self.available_tools.get_prompt(template)
            if not prompt_data:
                raise jinja2.TemplateNotFound(template)
            source = prompt_data.prompt or ""
            # Return: (source, filename, uptodate_func)
            return source, None, lambda: True
        except jinja2.TemplateNotFound:
            raise
        except (BadToolNameError, KeyError, AttributeError, FileNotFoundError):
            raise jinja2.TemplateNotFound(template)


def env_function(var_name: str, default: Optional[str] = None, required: bool = True) -> str:
    """Jinja2 function to access environment variables.

    Args:
        var_name: Name of environment variable
        default: Default value if not found
        required: If True, raises error when not found and no default

    Returns:
        Environment variable value or default

    Raises:
        LookupError: If required var not found

    Examples:
        {{ env('LOG_DIR') }}
        {{ env('OPTIONAL_VAR', 'default_value') }}
        {{ env('OPTIONAL_VAR', required=False) }}
    """
    value = os.getenv(var_name, default)
    if value is None and required:
        raise LookupError(f"Required environment variable {var_name} not found!")
    return value or ""


class _DataFirstEnvironment(jinja2.Environment):
    """Jinja environment that prefers mapping item access for ``a.b``.

    In stock Jinja, ``foo.bar`` tries ``getattr(foo, 'bar')`` before
    ``foo['bar']``. For a data-passing DSL that means dict keys named after
    dict methods (``items``, ``keys``, ``values``, ``get``, ``copy``, ...)
    resolve to the *method* instead of the data, e.g. ``outputs.x.items``
    returns the ``dict.items`` builtin. This subclass flips the order so
    mapping keys win, falling back to attribute access for objects. A mapping
    is resolved by key *only*: a missing key is undefined rather than a bound
    dict method, so absent data reads as undefined (falsy) in templates and
    ``if:`` conditions. For keys that are not dict methods the behaviour is
    identical to stock Jinja.
    """

    def getattr(self, obj: Any, attribute: str) -> Any:  # noqa: N802 - Jinja API
        # Mappings resolve by key only: a present key wins over any same-named
        # attribute (the data-first goal), and a *missing* key is undefined --
        # never a dict method (items/keys/values/get/...) -- so absent data
        # reads as undefined instead of a truthy bound method.
        if isinstance(obj, Mapping):
            if attribute in obj:
                return obj[attribute]
            return self.undefined(obj=obj, name=attribute)
        # Non-mappings keep stock-like resolution: item access, then attribute.
        try:
            return obj[attribute]
        except (TypeError, LookupError):
            pass
        try:
            return getattr(obj, attribute)
        except AttributeError:
            return self.undefined(obj=obj, name=attribute)


def create_jinja_environment(available_tools: "AvailableTools") -> jinja2.Environment:
    """Create configured Jinja2 environment for taskflow templates.

    Args:
        available_tools: AvailableTools instance for prompt loading

    Returns:
        Configured Jinja2 Environment
    """
    env = _DataFirstEnvironment(
        loader=PromptLoader(available_tools),
        # Use same delimiters as custom system
        variable_start_string='{{',
        variable_end_string='}}',
        block_start_string='{%',
        block_end_string='%}',
        # Disable auto-escaping (YAML context doesn't need HTML escaping)
        autoescape=False,
        # Keep whitespace for prompt formatting
        trim_blocks=True,
        lstrip_blocks=True,
        # Raise errors for undefined variables
        undefined=jinja2.StrictUndefined,
    )

    # Register custom functions
    env.globals['env'] = env_function

    return env


def render_template(
    template_str: str,
    available_tools: "AvailableTools",
    globals_dict: Optional[Dict[str, Any]] = None,
    inputs_dict: Optional[Dict[str, Any]] = None,
    result_value: Optional[Any] = None,
    outputs_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """Render a template string with provided context.

    Args:
        template_str: Template string to render
        available_tools: AvailableTools instance
        globals_dict: Global variables dict
        inputs_dict: Input variables dict
        result_value: Result value for repeat_prompt
        outputs_dict: Named task outputs, exposed as ``outputs.<id>``

    Returns:
        Rendered template string

    Raises:
        jinja2.TemplateError: On template rendering errors

    Examples:
        # Render with globals
        render_template("{{ globals.fruit }}", tools, globals_dict={'fruit': 'apple'})

        # Render with result
        render_template("{{ result.name }}", tools, result_value={'name': 'test'})

        # Render with a named task output
        render_template(
            "{{ outputs.list_functions.functions }}",
            tools,
            outputs_dict={'list_functions': {'functions': [...]}},
        )
    """
    jinja_env = create_jinja_environment(available_tools)

    # Build template context
    context = _build_context(globals_dict, inputs_dict, outputs_dict, result_value)

    # Render template
    template = jinja_env.from_string(template_str)
    return template.render(**context)


def evaluate_expression(
    expression: str,
    available_tools: "AvailableTools",
    globals_dict: Optional[Dict[str, Any]] = None,
    inputs_dict: Optional[Dict[str, Any]] = None,
    outputs_dict: Optional[Dict[str, Any]] = None,
    result_value: Optional[Any] = None,
) -> Any:
    """Evaluate a Jinja expression against the template context.

    Unlike :func:`render_template`, this returns the *actual Python object*
    the expression evaluates to (e.g. a list), not its string rendering. Used
    for the ``over:`` iterable selector so a typed list can be iterated
    directly instead of being round-tripped through JSON.

    The expression may be given bare (``outputs.foo.items``) or wrapped in
    ``{{ ... }}``; the wrapping is stripped for convenience.

    Raises:
        jinja2.TemplateError: On compilation or evaluation errors.
    """
    expr = expression.strip()
    if expr.startswith("{{") and expr.endswith("}}"):
        expr = expr[2:-2].strip()
    jinja_env = create_jinja_environment(available_tools)
    context = _build_context(globals_dict, inputs_dict, outputs_dict, result_value)
    compiled = jinja_env.compile_expression(expr)
    return compiled(**context)


def _build_context(
    globals_dict: Optional[Dict[str, Any]],
    inputs_dict: Optional[Dict[str, Any]],
    outputs_dict: Optional[Dict[str, Any]],
    result_value: Optional[Any],
) -> Dict[str, Any]:
    """Assemble the shared Jinja context for rendering and expressions."""
    context: Dict[str, Any] = {
        "globals": globals_dict or {},
        "inputs": inputs_dict or {},
        "outputs": outputs_dict or {},
    }
    if result_value is not None:
        context["result"] = result_value
    return context
