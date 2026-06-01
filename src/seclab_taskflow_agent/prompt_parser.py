# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Legacy argparse-based prompt parser.

When an agent has no explicit agents list, this module parses the user
prompt text to extract ``-p personality_name`` flags embedded in the
prompt itself.  The Typer-based CLI in :mod:`~seclab_taskflow_agent.cli`
has superseded this for normal invocations, but the parser is still used
at runtime by :mod:`~seclab_taskflow_agent.runner` and in several tests.
"""

from __future__ import annotations

import argparse
import logging

from .available_tools import AvailableTools

__all__ = ["parse_prompt_args"]


def parse_prompt_args(
    available_tools: AvailableTools, user_prompt: str | None = None
) -> tuple[str | None, str | None, bool, dict[str, str], str, str] | tuple[None, None, None, None, str, str]:
    """Legacy CLI parser kept for backwards compatibility with tests.

    Returns:
        Tuple of (personality, taskflow, list_models, cli_globals, prompt, help_msg).
    """
    parser = argparse.ArgumentParser(add_help=False, description="SecLab Taskflow Agent")
    parser.prog = ""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", help="The personality to use (mutex with -t)", required=False)
    group.add_argument("-t", help="The taskflow to use (mutex with -p)", required=False)
    group.add_argument("-l", help="List available tool call models and exit", action="store_true", required=False)
    parser.add_argument(
        "-g",
        "--global",
        dest="globals",
        action="append",
        help="Set global variable (KEY=VALUE). Can be used multiple times.",
        required=False,
    )
    parser.add_argument("prompt", nargs=argparse.REMAINDER)

    help_msg = parser.format_help()
    help_msg += "\nExamples:\n\n"
    help_msg += "`-p seclab_taskflow_agent.personalities.assistant explain modems to me please`\n"
    help_msg += "`-t examples.taskflows.example_globals -g fruit=apples`\n"
    try:
        args = parser.parse_known_args(user_prompt.split(" ") if user_prompt else None)
    except SystemExit as e:
        if e.code == 2:
            logging.exception("User provided incomplete prompt: %s", user_prompt)
        return None, None, None, None, "", help_msg
    except Exception:
        logging.exception("Failed to parse prompt: %s", user_prompt)
        return None, None, None, None, "", help_msg
    p = args[0].p.strip() if args[0].p else None
    t = args[0].t.strip() if args[0].t else None
    list_models = args[0].l

    cli_globals: dict[str, str] = {}
    if args[0].globals:
        for g in args[0].globals:
            if "=" not in g:
                logging.error("Invalid global variable format: %s. Expected KEY=VALUE", g)
                return None, None, None, None, "", help_msg
            key, value = g.split("=", 1)
            cli_globals[key.strip()] = value.strip()

    return p, t, list_models, cli_globals, " ".join(args[0].prompt), help_msg
