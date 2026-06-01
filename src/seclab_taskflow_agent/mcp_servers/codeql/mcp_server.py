# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

import csv
import json
import logging
import re
from pathlib import Path

# from mcp.server.fastmcp import FastMCP, Context
from fastmcp import FastMCP  # use FastMCP 2.0
from pydantic import Field

from seclab_taskflow_agent.path_utils import log_file_name, mcp_data_dir
from seclab_taskflow_agent.error_utils import error_with_message

from .client import _debug_log, file_from_uri, list_src_files, run_query, search_in_src_archive

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file_name("mcp_codeql.log"),
    filemode="a",
)

mcp = FastMCP("CodeQL")

CODEQL_DBS_BASE_PATH = mcp_data_dir("seclab-taskflow-agent", "codeql", "CODEQL_DBS_BASE_PATH")

# tool name -> templated query lookup for supported languages
TEMPLATED_QUERY_PATHS = {
    # to add a language, port the templated query pack and add its definition here
    "cpp": {
        "call_graph_to": "queries/mcp-cpp/call_graph_to.ql",
        "call_graph_from": "queries/mcp-cpp/call_graph_from.ql",
        "call_graph_from_to": "queries/mcp-cpp/call_graph_from_to.ql",
        "definition_location_for_function": "queries/mcp-cpp/definition_location_for_function.ql",
        "declaration_location_for_variable": "queries/mcp-cpp/declaration_location_for_variable.ql",
        "list_functions": "queries/mcp-cpp/list_functions.ql",
        "stmt_location": "queries/mcp-cpp/stmt_location.ql",
        "absolute_to_relative": "queries/mcp-cpp/absolute_to_relative.ql",
        "relative_to_absolute": "queries/mcp-cpp/relative_to_absolute.ql",
    },
    "javascript": {
        "call_graph_to": "queries/mcp-js/call_graph_to.ql",
        "call_graph_from": "queries/mcp-js/call_graph_from.ql",
        "definition_location_for_function": "queries/mcp-js/definition_location_for_function.ql",
        "absolute_to_relative": "queries/mcp-js/absolute_to_relative.ql",
        "relative_to_absolute": "queries/mcp-js/relative_to_absolute.ql",
    },
}


def _resolve_query_path(language: str, query: str) -> Path:
    global TEMPLATED_QUERY_PATHS
    if language not in TEMPLATED_QUERY_PATHS:
        raise error_with_message(RuntimeError, f"Error: Language `{language}` not supported!")
    query_path = TEMPLATED_QUERY_PATHS[language].get(query)
    if not query_path:
        raise error_with_message(RuntimeError, f"Error: query `{query}` not supported for `{language}`!")
    return Path(query_path)


def _resolve_db_path(relative_db_path: str | Path):
    global CODEQL_DBS_BASE_PATH
    # path joins will return "/B" if "/A" / "////B" etc. as well
    # not windows compatible and probably needs additional hardening
    relative_db_path = str(relative_db_path).strip().lstrip("/")
    relative_db_path = Path(relative_db_path)
    absolute_path = CODEQL_DBS_BASE_PATH / relative_db_path
    if not absolute_path.is_dir():
        _debug_log(f"Database path not found: {absolute_path}")
        raise error_with_message(RuntimeError, f"Error: Database not found at {absolute_path}!")
    return absolute_path


# our query result format is: "human readable template {val0} {val1},'key0,key1',val0,val1"
def _csv_to_json_obj(raw):
    results = []
    reader = csv.reader(raw.strip().splitlines())
    try:
        for i, row in enumerate(reader):
            if i == 0:
                continue
            # col1 has what we care about, but offer flexibility
            keys = row[1].split(",")
            this_obj = {"description": row[0].format(*row[2:])}
            for j, k in enumerate(keys):
                this_obj[k.strip()] = row[j + 2]
            results.append(this_obj)
    except csv.Error as e:
        return ["Error: CSV parsing error: " + str(e)]
    return json.dumps(results, indent=2)


def _get_file_contents(db: str | Path, uri: str):
    """Retrieve file contents from a CodeQL database"""
    db = Path(db)
    return file_from_uri(uri, db)


def _run_query(query_name: str, database_path: str, language: str, template_values: dict):
    """Run a CodeQL query and return the results"""

    try:
        database_path = _resolve_db_path(database_path)
    except RuntimeError:
        return json.dumps([f"The database path for {database_path} could not be resolved"])
    try:
        query_path = _resolve_query_path(language, query_name)
    except RuntimeError:
        return json.dumps([f"The query {query_name} is not supported for language: {language}"])
    try:
        csv = run_query(
            Path(__file__).parent.resolve() / query_path,
            database_path,
            fmt="csv",
            template_values=template_values,
            log_stderr=True,
        )
        return _csv_to_json_obj(csv)
    except Exception as e:
        return json.dumps([f"The query {query_name} encountered an error: {e}"])


@mcp.tool()
def get_file_contents(
    file_uri: str = Field(
        description="The file URI to get contents for. The URI scheme is defined as `file://path` and `file://path:region`. Examples of file URI: `file:///path/to/file:1:2:3:4`, `file:///path/to/file`. File URIs optionally contain a region definition that looks like `start_line:start_column:end_line:end_column` which will limit the contents returned to the specified region, for example `file:///path/to/file:1:2:3:4` indicates a file region of `1:2:3:4` which would return the content of the file starting at line 1, column 1 and ending at line 3 column 4. Line and column indices are 1-based, meaning line and column values start at 1. If the region is ommitted the full contents of the file will be returned, for example `file:///path/to/file` returns the full contents of `/path/to/file`."
    ),
    database_path: str = Field(description="The CodeQL database path."),
):
    """Get the contents of a file URI from a CodeQL database path."""

    database_path = _resolve_db_path(database_path)
    try:
        # fix up any incorrectly formatted relative path uri
        if not file_uri.startswith("file:///"):
            file_uri = file_uri.removeprefix("file://")
            file_uri = "file:///" + file_uri.lstrip("/")
        results = _get_file_contents(database_path, file_uri)
    except Exception as e:
        results = f"Error: could not retrieve {file_uri}: {e}"
    return results


@mcp.tool()
def list_source_files(
    database_path: str = Field(description="The CodeQL database path."),
    regex_filter: str = Field(description="Optional Regex filter.", default=r"[\s\S]+"),
):
    """List the available source files in a CodeQL database using their file:// URI"""
    database_path = _resolve_db_path(database_path)
    results = list_src_files(database_path, as_uri=True)
    return json.dumps([{"uri": item} for item in results if re.search(regex_filter, item)], indent=2)


@mcp.tool()
def search_in_source_code(
    database_path: str = Field(description="The CodeQL database path."),
    search_term: str = Field(description="The term to search in the source code"),
):
    """
    Search for a string in the source code. Returns the line number and file.
    """
    resolved_database_path = _resolve_db_path(database_path)
    results = search_in_src_archive(resolved_database_path, search_term)
    out = []
    if isinstance(results, dict):
        for k, v in results.items():
            out.append({"database": database_path, "path": k, "lines": v})
    return json.dumps(out, indent=2)


@mcp.tool()
def definition_location_for_function(
    target_definition: str = Field(
        description="The function to get the source code location file URI of its definition for."
    ),
    database_path: str = Field(description="The CodeQL database path."),
    language: str = Field(description="The language used for the CodeQL database."),
):
    """Return the location of a function definition. Returns the region of the function as a file URI."""
    return _run_query(
        "definition_location_for_function", database_path, language, {"targetDefinition": target_definition}
    )


@mcp.tool()
def declaration_location_for_variable(
    target_declaration: str = Field(
        description="The variable to get the source code location file URI of its declaration for."
    ),
    database_path: str = Field(description="The CodeQL database path."),
    language: str = Field(description="The language used for the CodeQL database."),
):
    """Return the location of a variable declaration. Returns the region of the variable, as well as its enclosing function as file URI."""
    return _run_query(
        "declaration_location_for_variable", database_path, language, {"targetDeclaration": target_declaration}
    )


@mcp.tool()
def statement_location(
    target_statement: str = Field(
        description="The type of statement to get the source code location file URI of its definition for."
    ),
    database_path: str = Field(description="The CodeQL database path."),
    language: str = Field(description="The language used for the CodeQL database."),
):
    """Return the location of a statement. Returns the region of the statement, as well as its enclosing function as file URI."""
    new_target_statement = target_statement + "%"
    return _run_query("stmt_location", database_path, language, {"targetStmt": new_target_statement})


@mcp.tool()
def call_graph_to(
    target_function: str = Field(description="The target function to get calls to."),
    database_path: str = Field(description="The CodeQL database path."),
    language: str = Field(description="The language used for the CodeQL database."),
):
    """Return function calls to a function with their locations."""
    return _run_query("call_graph_to", database_path, language, {"targetFunction": target_function})


@mcp.tool()
def call_graph_from(
    source_function: str = Field(description="The source function to get calls from."),
    database_path: str = Field(description="The CodeQL database path."),
    language: str = Field(description="The language used for the CodeQL database."),
):
    """Return calls from a function with their locations."""
    return _run_query("call_graph_from", database_path, language, {"sourceFunction": source_function})


@mcp.tool()
def call_graph_from_to(
    source_function: str = Field(description="The source function for the call path."),
    target_function: str = Field(description="The target function for the call path."),
    database_path: str = Field(description="The CodeQL database path."),
    language: str = Field(description="The language used for the CodeQL database."),
):
    """Determine if a call path between a source function and a target function exists."""
    return _run_query(
        "call_graph_from_to",
        database_path,
        language,
        {"sourceFunction": source_function, "targetFunction": target_function},
    )


@mcp.tool()
def list_functions(
    database_path: str = Field(description="The CodeQL database path."),
    language: str = Field(description="The language used for the CodeQL database."),
):
    """List all functions and their locations in a CodeQL database."""
    return _run_query("list_functions", database_path, language, {})


if __name__ == "__main__":
    mcp.run(show_banner=False, transport="http", host="127.0.0.1", port=9999)
