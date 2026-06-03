# SPDX-FileCopyrightText: GitHub, Inc.
# SPDX-License-Identifier: MIT

# a query-server2 codeql client
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

from seclab_taskflow_agent.path_utils import log_file_name

# this is a local fork of https://github.com/riga/jsonrpyc modified for our purposes
from . import jsonrpyc

WAIT_INTERVAL = 0.1


# for when our stdout goes into the void
def _debug_log(msg):
    with open("codeql-debug.log", "a+") as f:
        f.write(msg + "\n")


def shell_command_to_string(cmd):
    print(f"Executing: {cmd}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    stdout, stderr = p.communicate()
    p.wait()
    if p.returncode:
        raise RuntimeError(stderr)
    return stdout


class CodeQL:
    def __init__(
        self,
        codeql_cli=os.getenv("CODEQL_CLI", default="codeql"),
        server_options=["--threads=0", "--quiet"],
        log_stderr=False,
    ):
        self.server_options = server_options.copy()
        if log_stderr:
            self.stderr_log = log_file_name("codeql_stderr_log.log")
            self.server_options.append("--log-to-stderr")
        else:
            self.stderr_log = os.devnull
        self.codeql_cli = codeql_cli.split()
        self.search_paths = []
        self.active_database = None
        self.active_connection = None
        self.active_query_id = None
        self.active_query_error = (False, "")
        self.progress_id = 0
        # clients can override e.g. the default ql/progressUpdated callback if they wish
        self.method_handlers = {}

    # def __del__(self):
    #     self._server_stop()

    # server state management
    def _server_resolve_ram(self, max_ram=0):
        max_ram_arg = [f"-M={max_ram}"] if max_ram else []
        return shell_command_to_string(self.codeql_cli + ["resolve", "ram"] + max_ram_arg + ["--"]).strip().split("\n")

    def _server_start(self):
        ram_options = self._server_resolve_ram()
        server_cmd = ["execute", "query-server2"]
        server_cmd += ram_options
        server_cmd += self.server_options
        self.stderr_log = open(self.stderr_log, "a")
        p = subprocess.Popen(
            self.codeql_cli + server_cmd,
            text=True,
            bufsize=1,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_log,
        )

        # set some default callbacks for common notifications
        def _handle_ql_progressUpdated(params):
            print(f">> Progress: {params.get('step')}/{params.get('maxStep')} status: {params.get('message')}")

        ql_progressUpdated = "ql/progressUpdated"
        if ql_progressUpdated not in self.method_handlers:
            self.method_handlers[ql_progressUpdated] = _handle_ql_progressUpdated

        rpc = jsonrpyc.RPC(method_handlers=self.method_handlers, stdout=p.stdin, stdin=p.stdout)
        self.active_connection = (p, rpc)

    def _server_stop(self):
        # no need to deregister, just close stdin for graceful exit of query-server2
        # if self.active_database:
        #     self._server_deregister_database(self.active_database)
        #     max_wait = 5
        #     while self.active_database and max_wait:
        #         # the server might be unresponsive, but give it a fighting chance
        #         time.sleep(1)
        #         max_wait -= 1
        if self.active_connection:
            p, rpc = self.active_connection
            try:
                p.stdin.close()
                p.stdout.close()
                if p.stderr:
                    p.stderr.close()
            except (OSError, ValueError):
                pass
            p.wait()
        self.active_database = None
        # deletion of rpc object also triggers the thread cleanup for watchdog
        self.active_connection = None
        self.active_query_id = None
        self.active_query_error = (False, "")

    def _server_connection_ready_p(self):
        return True if self.active_connection else False

    def _server_next_progress_id(self):
        self.progress_id += 1
        return self.progress_id

    def _server_rpc_call(self, method, params, callback=None):
        p, rpc = self.active_connection
        return rpc(method, params=params, callback=callback, block=-1 if callback else 0.1)

    def _server_rpc_notify(self, method, params):
        p, rpc = self.active_connection
        return rpc(method, params=params, callback=None, block=-1)

    def _server_register_database(self, database_path):
        if self.active_database:
            self._server_deregister_database(self.active_database)
            while self.active_database:
                time.sleep(WAIT_INTERVAL)
        database = self._database_info(database_path)
        database["path"] = str(Path(database_path).resolve())
        rpc_method = "evaluation/registerDatabases"

        def _callback(err: Exception, res: str | None = None):
            if err:
                raise err
            self.active_database = database
            print(f"++ {rpc_method}: {res}")

        return self._server_rpc_call(
            rpc_method,
            {
                "progressId": self._server_next_progress_id(),
                "body": {"databases": [str(Path(database_path).resolve())]},
            },
            callback=_callback,
        )

    def _server_deregister_database(self, database):
        rpc_method = "evaluation/deregisterDatabases"

        def _callback(err: Exception, res: str | None = None):
            if err:
                raise err
            self.active_database = None
            print(f"++ {rpc_method}: {res}")

        return self._server_rpc_call(
            rpc_method,
            {"progressId": self._server_next_progress_id(), "body": {"databases": [database["path"]]}},
            callback=_callback,
        )

    def _server_active_database(self):
        return self.active_database

    def _server_cancel_active_query(self):
        if self.active_query_id:
            rpc_method = "$/cancelRequest"
            self._server_rpc_notify(rpc_method, params={"id": self.active_query_id})
            self.active_query_id = None

    def _server_request_run(
        self,
        bqrs_path,
        query_path,
        library_paths,
        quick_eval_pos: dict | None = None,
        template_values: dict | None = None,
    ):
        if not self.active_database:
            raise RuntimeError("No Active Database")

        if not self.active_connection:
            raise RuntimeError("No Active Connection")

        if isinstance(quick_eval_pos, dict):
            # A quick eval position contains:
            # fileName
            # line
            # column
            # endLine
            # endColumn
            query_target = {"quickEval": {"quickEvalPos": quick_eval_pos}}
        else:
            query_target = {"query": {"xx": ""}}

        query_params = {
            "body": {
                "db": self.active_database["path"],
                "additionalPacks": ":".join(library_paths),
                "singletonExternalInputs": template_values if template_values else {},
                "outputPath": str(bqrs_path),
                "queryPath": str(query_path),
                "target": query_target,
            }
        }

        rpc_method = "evaluation/runQuery"

        def _callback(err: Exception, res: str | None = None):
            def _check_runquery_result_for_errors(params: dict):
                if "resultType" in params and "message" in params:
                    result_type = params["resultType"]
                    message = params["message"]
                    match result_type:
                        case 0:
                            return False, ""
                        case 1:
                            print(f"xx ERROR Other: {message}")
                            return True, message
                        case 2:
                            print(f"xx ERROR Compilation: {message}")
                            return True, message
                        case 3:
                            print(f"xx ERROR OOM: {message}")
                            return True, message
                        case 4:
                            print(f"xx ERROR Query Canceled: {message}")
                            return True, message
                        case 5:
                            print(f"xx ERROR DB Scheme mismatch: {message}")
                            return True, message
                        case 6:
                            print(f"xx ERROR DB Scheme no upgrade found: {message}")
                            return True, message
                        case _:
                            print(f"xx ERROR: unknown result type {result_type}: {message}")
                            return True, message
                else:
                    return False, ""

            if isinstance(res, dict):
                self.active_query_error = _check_runquery_result_for_errors(res)
            else:
                self.active_query_error = (True, f"Unknown result state: {res}")
            self.active_query_id = None
            print(f"++ {rpc_method}: {res}")
            if err:
                raise err

        self.active_query_id = self._server_rpc_call(rpc_method, query_params, callback=_callback)
        return self.active_query_id

    def _server_run_query_from_path(self, bqrs_path, query_path, quick_eval_pos=None, template_values=None):
        library_paths = self._resolve_library_paths(query_path)
        return self._server_request_run(
            bqrs_path, query_path, library_paths, quick_eval_pos=quick_eval_pos, template_values=template_values
        )

    # utility functions
    def _search_path(self):
        return ":".join(self.search_paths)

    def _search_paths_from_codeql_config(self, config="~/.config/codeql/config"):
        try:
            with open(config) as f:
                match = re.search(r"^--search-path(\s+|=)\s*(.*)", f.read())
                if match and match.group(2):
                    return match.group(2).split(":")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        return []

    def _lang_server_contact(self):
        lsp_server_cmd = ["execute", "language-server"]
        lsp_server_cmd += [f"--search-path={self._search_path()}"] if self._search_path() else []
        lsp_server_cmd += ["--check-errors", "ON_CHANGE", "-q"]
        return self.codeql_cli + lsp_server_cmd

    def _get_cli_version(self):
        return shell_command_to_string(self.codeql_cli + ["version"])

    def _format(self, query):
        return shell_command_to_string(self.codeql_cli + ["query", "format", "--no-syntax=errors", "--", query])

    def _resolve_query_server(self):
        help_msg = shell_command_to_string(self.codeql_cli + ["excute", "--help"])
        if not re.search("query-server2", help_msg):
            raise RuntimeError("Legacy server not supported!")
        return "query-server2"

    def _resolve_library_paths(self, query_path):
        search_path = self._search_path()
        args = ["resolve", "library-path"]
        args += ["-v", "--log-to-stderr", "--format=json"]
        if search_path:
            print(f"Using search path: {search_path}")
            args += [f'--additional-packs="{search_path}"']
        args += [f"--query={query_path}"]
        return json.loads(shell_command_to_string(self.codeql_cli + args))

    def _resolve_qlpack_paths(self, query_dir):
        return json.loads(
            shell_command_to_string(
                self.codeql_cli
                + ["resolve", "qlpacks", "-v", "--log-to-stderr", "--format=json", f"--search-path={query_dir}"]
            )
        )

    def _database_info(self, database_path):
        return json.loads(
            shell_command_to_string(
                self.codeql_cli
                + ["resolve", "database", "-v", "--log-to-stderr", "--format=json", "--", f"{database_path}"]
            )
        )

    def _database_upgrades(self, database_scheme):
        return json.loads(
            shell_command_to_string(
                self.codeql_cli
                + ["resolve", "upgrades", "-v", "--log-to-stderr", "--format=json", f"--dbscheme={database_scheme}"]
            )
        )

    def _query_info(self, query_path):
        return json.loads(
            shell_command_to_string(
                self.codeql_cli
                + ["resolve", "metadata", "-v", "--log-to-stderr", "--format=json", "--", f"{query_path}"]
            )
        )

    def _bqrs_info(self, bqrs_path):
        return json.loads(
            shell_command_to_string(
                self.codeql_cli + ["bqrs", "info", "-v", "--log-to-stderr", "--format=json", "--", f"{bqrs_path}"]
            )
        )

    def _bqrs_to_csv(self, bqrs_path, entities=""):
        csv_out = Path(bqrs_path).with_suffix(".csv")
        args = ["bqrs", "decode", f"--output={csv_out}", "--format=csv"]
        args += [f"--entities={entities}"] if entities else []
        args += ["--", f"{bqrs_path}"]
        try:
            shell_command_to_string(self.codeql_cli + args)
            with open(csv_out) as f:
                return f.read()
        except RuntimeError as e:
            print(f"Could not decode {bqrs_path} to {csv_out}: {e}")
            return ""

    def _bqrs_to_json(self, bqrs_path, entities):
        json_out = Path(bqrs_path).with_suffix(".json")
        args = ["bqrs", "decode", f"--output={json_out}", "--format=json"]
        args += [f"--entities={entities}"] if entities else []
        args += ["--", f"{bqrs_path}"]
        try:
            shell_command_to_string(self.codeql_cli + args)
            with open(json_out) as f:
                return f.read()
        except RuntimeError as e:
            print(f"Could not decode {bqrs_path} to {json_out}: {e}")
            return ""

    def _bqrs_to_sarif(self, bqrs_path, query_info, max_paths=10):
        sarif_out = Path(bqrs_path).with_suffix(".sarif")
        if shell_command_to_string(
            self.codeql_cli
            + [
                "bqrs",
                "interpret",
                "-v",
                "--log-to-stderr",
                f"-t=id={query_info.get('id')}",
                f"-t=kind={query_info.get('kind')}",
                f"--output={sarif_out}",
                "--format=sarif-latest",
                f"--max-paths={max_paths}",
                "--no-group-results",
                "--",
                f"{bqrs_path}",
            ]
        ):
            with open(sarif_out) as f:
                return f.read()
        print(f"Could not decode {bqrs_path} to {sarif_out}")
        return ""


class QueryServer(CodeQL):
    def __init__(self, database: Path, keep_alive=False, log_stderr=False):
        super().__init__(log_stderr=log_stderr)
        self.database = database
        self.keep_alive = keep_alive

    def __enter__(self):
        global _ACTIVE_CODEQL_SERVERS
        if self.database in _ACTIVE_CODEQL_SERVERS:
            return _ACTIVE_CODEQL_SERVERS[self.database]
        if not self.active_connection:
            self._server_start()
        print("Waiting for server start ...")
        while not self.active_connection:
            time.sleep(WAIT_INTERVAL)
        if not self.active_database:
            self._server_register_database(self.database)
        print("Waiting for database registration ...")
        while not self.active_database:
            time.sleep(WAIT_INTERVAL)
        if self.keep_alive:
            _ACTIVE_CODEQL_SERVERS[self.database] = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.database not in _ACTIVE_CODEQL_SERVERS:
            self._server_stop()


_ACTIVE_CODEQL_SERVERS: dict[Path, QueryServer] = {}


def get_query_position(query_path: str | Path, target: str):
    query_path = Path(query_path)
    lines = query_path.read_text().splitlines()
    pos = None
    for i, line in enumerate(lines):
        # the first occurrence of a predicate should be its definition?
        pattern = rf"\b({re.escape(target)})\s*\(" if not target[0].isupper() else rf"\bclass\s+({re.escape(target)})\b"
        if match := re.search(pattern, line):
            pos = {
                "fileName": str(query_path),
                "line": 1 + i,
                "column": 1 + match.start(1),
                "endLine": 1 + i,
                "endColumn": 1 + match.start(1) + len(target),
            }
            break
    return pos


def _file_uri_to_path(uri):
    # note: codeql file:// uris are always formatted as absolute paths
    # otherwise on parse the first component would be considered netloc
    # so even for relative paths ALWAYS use 'file://' + '/some/path'
    # internally the codeql client will resolve both relative and full paths
    # regardless of root directory differences
    if not uri.startswith("file:///"):
        raise ValueError("URI path should be formatted as absolute")
    # note: don't try to parse paths like "file://a/b" because that returns "/b", should be "file:///a/b"
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Not a file:// uri: {uri}")
    path = unquote(parsed.path)
    region = None
    if ":" in path:
        path, start_line, start_col, end_line, end_col = path.split(":")
        region = (abs(int(start_line)), abs(int(start_col)), abs(int(end_line)), abs(int(end_col)))
    return path, region


def _get_source_prefix(database_path: Path, strip_leading_slash=True) -> str:
    # grab the source prefix from codeql-database.yml
    db_yml_path = Path(database_path) / Path("codeql-database.yml")
    with open(db_yml_path) as stream:
        try:
            # normalize
            source_prefix = "/" + yaml.safe_load(stream)["sourceLocationPrefix"].strip().strip("/") + "/"
            if strip_leading_slash:
                source_prefix = source_prefix.lstrip("/")
            return source_prefix
        except (yaml.YAMLError, FileNotFoundError, KeyError) as e:
            logging.error("Error parsing sourceLocationPrefix: %s", e)
            raise


def list_src_files(database_path: str | Path, as_uri=False, strip_prefix=True):
    src_path = Path(database_path) / Path("src.zip")
    files = shell_command_to_string(["zipinfo", "-1", src_path]).split("\n")
    source_prefix = _get_source_prefix(Path(database_path))
    # file:// uri are formatted absolute paths even if they're relative
    files = [
        f"{'file:///' if as_uri else ''}{path.strip().removeprefix(source_prefix if strip_prefix else '')}"
        for path in files
    ]
    return files


def search_in_src_archive(database_path: str, search_term: str, as_uri=False, strip_prefix=True):
    database_path = Path(database_path)
    src_path = database_path / Path("src.zip")
    results = {}
    source_prefix = _get_source_prefix(database_path)
    with zipfile.ZipFile(src_path) as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            with z.open(entry, "r") as f:
                for i, line in enumerate(f):
                    if search_term in str(line):
                        path = entry.filename.strip().removeprefix(source_prefix if strip_prefix else "")
                        path = f"{'file:///' if as_uri else ''}{path}"
                        if path not in results:
                            results[path] = [i + 1]
                        else:
                            results[path].append(i + 1)
    return results


def _file_from_src_archive(relative_path: str | Path, database_path: str | Path, region: tuple | None = None):
    # our shell utility is Popen based, so no expansions occur
    database_path = Path(database_path)
    src_path = database_path / Path("src.zip")
    source_prefix = _get_source_prefix(Path(database_path))
    # normalize relative path
    relative_path = Path(str(relative_path).lstrip("/").removeprefix(source_prefix))
    resolved_path = Path(source_prefix) / Path(relative_path)
    files = list_src_files(database_path, as_uri=False, strip_prefix=False)
    # fall back to relative path if resolved_path does not exist (might be a build dep file)
    if str(resolved_path) not in files:
        resolved_path = Path(relative_path)
    file_data = shell_command_to_string(["unzip", "-p", src_path, f"{resolved_path!s}"])
    if region:

        def region_from_file():
            # regions are 1+ based and look like 1:2:3:4
            # 0 values indicate we want the maximum available
            lines = file_data.split("\n")
            start_line, start_col, end_line, end_col = region
            start_line -= 1 if start_line else 0
            start_col -= 1 if start_col else 0
            end_line -= 1 if end_line else 0
            end_col -= 1 if end_col else 0
            region_data = ""
            if not end_line:
                end_line = len(lines) - 1
            i = start_line
            while i <= end_line:
                if start_line == i:
                    if start_line == end_line:
                        if start_col and end_col:
                            region_data += lines[start_line][start_col : end_col + 1]
                        elif start_col:
                            region_data += lines[start_line][start_col:] + "\n"
                        elif end_col:
                            region_data += lines[start_line][: end_col + 1]
                        else:
                            region_data += lines[start_line] + "\n"
                    else:
                        region_data += lines[start_line][start_col:] + "\n"
                elif end_line == i:
                    if start_line != end_line:
                        if end_col:
                            region_data += lines[end_line][: end_col + 1]
                        else:
                            region_data += lines[end_line] + "\n"
                else:
                    region_data += lines[i] + "\n"
                i += 1
            return region_data

        file_data = region_from_file()
    return file_data


def file_from_uri(uri: str, database_path: str | Path):
    path, region = _file_uri_to_path(uri)
    return _file_from_src_archive(path, database_path, region=region)


def run_query(
    query_path: str | Path,
    database: Path,
    entities="string",
    fmt="json",
    search_paths=[],
    # a quick eval predicate or class name
    target="",
    progress_callback=None,
    template_values=None,
    # keep the query server alive if desired
    keep_alive=True,
    log_stderr=False,
):
    result = ""
    query_path = Path(query_path)
    target_pos = None
    if target:
        target_pos = get_query_position(query_path, target)
        if not target_pos:
            raise ValueError(f"Could not resolve quick eval target for {target}")
    try:
        with (
            QueryServer(database, keep_alive=keep_alive, log_stderr=log_stderr) as server,
            tempfile.TemporaryDirectory() as base_path,
        ):
            if callable(progress_callback):
                server.method_handlers["ql/progressUpdated"] = progress_callback
            bqrs_path = base_path / Path("query.bqrs")
            if search_paths:
                server.search_paths += search_paths

            server._server_run_query_from_path(
                bqrs_path, query_path, quick_eval_pos=target_pos, template_values=template_values
            )
            while server.active_query_id:
                time.sleep(WAIT_INTERVAL)
            failed, msg = server.active_query_error
            if failed:
                raise RuntimeError(msg)
            match fmt:
                case "json":
                    result = server._bqrs_to_json(bqrs_path, entities=entities)
                case "csv":
                    result = server._bqrs_to_csv(bqrs_path, entities=entities)
                case "sarif":
                    result = server._bqrs_to_sarif(bqrs_path, server._query_info(query_path))
                case _:
                    raise ValueError("Unsupported output format {fmt}")
    except Exception as e:
        raise RuntimeError(f"Error in run_query: {e}") from e
    return result
