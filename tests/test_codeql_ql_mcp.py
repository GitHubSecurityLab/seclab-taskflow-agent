# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the codeql_ql_mcp toolbox configuration.

Validates that the codeql_ql_mcp toolbox YAML loads correctly and configures
the codeql-development-mcp-server as a drop-in replacement for the legacy
codeql toolbox.
"""

import yaml

from seclab_taskflow_agent.models import ToolboxDocument

TOOLBOX_PATH = "src/seclab_taskflow_agent/toolboxes/codeql_ql_mcp.yaml"

# All 10 CodeQL-supported language acronyms
CODEQL_LANGUAGES = [
    "actions",
    "cpp",
    "csharp",
    "go",
    "java",
    "javascript",
    "python",
    "ruby",
    "rust",
    "swift",
]

# Environment variables the ql-mcp server needs to receive
REQUIRED_ENV_VARS = [
    "CODEQL_PATH",
    "CODEQL_DATABASES_BASE_DIRS",
    "ENABLE_ANNOTATION_TOOLS",
]


def _load_toolbox() -> ToolboxDocument:
    with open(TOOLBOX_PATH) as f:
        data = yaml.safe_load(f)
    return ToolboxDocument(**data)


class TestCodeqlQlMcpToolbox:
    """Validate the codeql_ql_mcp toolbox YAML."""

    def test_parses_into_valid_toolbox_document(self):
        doc = _load_toolbox()
        assert doc.header.filetype == "toolbox"
        assert doc.header.version == "1.0"

    def test_transport_is_stdio(self):
        doc = _load_toolbox()
        assert doc.server_params.kind == "stdio"

    def test_command_is_ql_mcp_binary(self):
        doc = _load_toolbox()
        assert doc.server_params.command == "codeql-development-mcp-server"

    def test_env_maps_seclab_vars_to_ql_mcp_vars(self):
        doc = _load_toolbox()
        env = doc.server_params.env
        assert env is not None
        for var in REQUIRED_ENV_VARS:
            assert var in env, f"Missing required env var: {var}"
        # Verify the mappings from seclab env names to ql-mcp env names
        assert env["CODEQL_PATH"] == "{{ env('CODEQL_CLI') }}"
        assert env["CODEQL_DATABASES_BASE_DIRS"] == "{{ env('CODEQL_DBS_BASE_PATH') }}"
        assert env["ENABLE_ANNOTATION_TOOLS"] == "true"

    def test_server_prompt_lists_all_languages(self):
        doc = _load_toolbox()
        prompt = doc.server_prompt
        assert prompt != ""
        for lang in CODEQL_LANGUAGES:
            assert lang in prompt, f"Language '{lang}' missing from server_prompt"

    def test_server_prompt_has_file_uri_docs(self):
        doc = _load_toolbox()
        assert "file://" in doc.server_prompt

    def test_annotation_tools_enabled(self):
        doc = _load_toolbox()
        assert doc.server_params.env["ENABLE_ANNOTATION_TOOLS"] == "true"

    def test_server_prompt_covers_critical_tools(self):
        """Verify key tools that seclab-taskflow use cases depend on."""
        doc = _load_toolbox()
        prompt = doc.server_prompt
        critical_tools = [
            "read_database_source",
            "list_codeql_databases",
            "register_database",
            "codeql_query_run",
            "codeql_database_analyze",
            "audit_store_findings",
            "audit_list_findings",
            "audit_add_notes",
            "audit_clear_repo",
            "sarif_list_rules",
            "search_ql_code",
            "quick_evaluate",
        ]
        for tool in critical_tools:
            assert tool in prompt, f"Critical tool '{tool}' missing from server_prompt"
