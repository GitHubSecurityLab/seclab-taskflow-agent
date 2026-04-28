# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Jinja2 template utilities for taskflow template rendering."""

import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import jinja2

__all__ = ["PromptLoader", "create_jinja_environment", "env_function", "render_template"]

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


def create_jinja_environment(available_tools: "AvailableTools") -> jinja2.Environment:
    """Create configured Jinja2 environment for taskflow templates.

    Args:
        available_tools: AvailableTools instance for prompt loading

    Returns:
        Configured Jinja2 Environment
    """
    env = jinja2.Environment(
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
) -> str:
    """Render a template string with provided context.

    Args:
        template_str: Template string to render
        available_tools: AvailableTools instance
        globals_dict: Global variables dict
        inputs_dict: Input variables dict
        result_value: Result value for repeat_prompt

    Returns:
        Rendered template string

    Raises:
        jinja2.TemplateError: On template rendering errors

    Examples:
        # Render with globals
        render_template("{{ globals.fruit }}", tools, globals_dict={'fruit': 'apple'})

        # Render with result
        render_template("{{ result.name }}", tools, result_value={'name': 'test'})

        # Render with all context types
        render_template(
            "{{ globals.x }} {{ inputs.y }} {{ result.z }}",
            tools,
            globals_dict={'x': 1},
            inputs_dict={'y': 2},
            result_value={'z': 3}
        )
    """
    jinja_env = create_jinja_environment(available_tools)

    # Build template context
    context = {
        'globals': globals_dict or {},
        'inputs': inputs_dict or {},
    }

    # Add result if provided
    if result_value is not None:
        context['result'] = result_value

    # Render template
    template = jinja_env.from_string(template_str)
    return template.render(**context)
