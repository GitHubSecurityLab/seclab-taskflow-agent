import asyncio
from threading import Thread
import argparse
import os
import sys
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from pprint import pprint, pformat
import re
import json
import uuid
import pathlib

from .agent import DEFAULT_MODEL, TaskRunHooks, TaskAgentHooks
#from agents.run import DEFAULT_MAX_TURNS # XXX: this is 10, we need more than that
from agents.exceptions import MaxTurnsExceeded, AgentsException
from agents.agent import ModelSettings
from agents.mcp import MCPServer, MCPServerStdio, MCPServerSse, MCPServerStreamableHttp, create_static_tool_filter
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents import Tool, RunContextWrapper, TContext, Agent
from openai import BadRequestError, APITimeoutError, RateLimitError
from openai.types.responses import ResponseTextDeltaEvent
from typing import Any

from .shell_utils import shell_tool_call
from .mcp_utils import DEFAULT_MCP_CLIENT_SESSION_TIMEOUT, ReconnectingMCPServerStdio, AsyncDebugMCPServerStdio, MCPNamespaceWrap, mcp_client_params, mcp_system_prompt, StreamableMCPThread, compress_name
from .render_utils import render_model_output, flush_async_output
from .env_utils import TmpEnv
from .yaml_parser import YamlParser
from .agent import TaskAgent
from .capi import list_tool_call_models
from .available_tools import AvailableTools

load_dotenv()

# only model output or help message should go to stdout, everything else goes to log
logging.getLogger('').setLevel(logging.NOTSET)

# Create logs directory if it doesn't exist
# This is needed when running the code as a package from any directory
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'task_agent.log'),
    maxBytes=1024*1024*10,
    backupCount=10)
log_file_handler.setLevel(os.getenv('TASK_AGENT_LOGLEVEL', default='DEBUG'))
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(log_file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)  # log only ERROR and above to console
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logging.getLogger('').addHandler(console_handler)

DEFAULT_MAX_TURNS = 50
RATE_LIMIT_BACKOFF = 5
MAX_RATE_LIMIT_BACKOFF = 120
MAX_API_RETRY = 5
MCP_CLEANUP_TIMEOUT = 5

def parse_prompt_args(available_tools: AvailableTools,
                      user_prompt: str | None = None):
    parser = argparse.ArgumentParser(add_help=False, description="SecLab Taskflow Agent")
    parser.prog = ''
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", help="The personality to use (mutex with -t)", required=False)
    group.add_argument("-t", help="The taskflow to use (mutex with -p)", required=False)
    group.add_argument("-l", help="List available tool call models and exit", action='store_true', required=False)
    parser.add_argument('prompt', nargs=argparse.REMAINDER)
    #parser.add_argument('remainder', nargs=argparse.REMAINDER, help="Remaining args")
    help_msg = parser.format_help()
    help_msg += "\nAvailable Personalities:\n\n"
    for k in available_tools.personalities:
        help_msg += f"`{k}`\n"
    help_msg += "\nAvailable Taskflows:\n\n"
    for k in available_tools.taskflows:
        help_msg += f"`{k}`\n"
    help_msg += "\nExamples:\n\n"
    help_msg += "`-p assistant explain modems to me please`\n"
    try:
        args = parser.parse_known_args(user_prompt.split(' ') if user_prompt else None)
    except SystemExit as e:
        if e.code == 2:
            logging.error(f"User provided incomplete prompt: {user_prompt}")
            return None, None, None, help_msg
    p = args[0].p.strip() if args[0].p else None
    t = args[0].t.strip() if args[0].t else None
    l = args[0].l
    return p, t, l, ' '.join(args[0].prompt), help_msg

async def deploy_task_agents(available_tools: AvailableTools,
                             agents: dict,
                             prompt: str,
                             async_task: bool = False,
                             toolboxes_override: list = [],
                             blocked_tools: list = [],
                             headless: bool = False,
                             exclude_from_context: bool = False,
                             max_turns: int = DEFAULT_MAX_TURNS,
                             model: str = DEFAULT_MODEL,
                             model_settings: ModelSettings | None = None,
                             run_hooks: TaskRunHooks | None = None,
                             agent_hooks: TaskAgentHooks | None = None):

    task_id = str(uuid.uuid4())
    await render_model_output(f"** 🤖💪 Deploying Task Flow Agent(s): {list(agents.keys())}\n")
    await render_model_output(f"** 🤖💪 Task ID: {task_id}\n")

    mcp_servers = []
    server_prompts = []
    toolboxes = []

    if toolboxes_override:
        # limit tools to task specified tools if set
        toolboxes = toolboxes_override
    else:
        # otherwise all agents have the disjunction of all their tools available
        for k, v in agents.items():
            if v.get('toolboxes', []):
                toolboxes += [tb for tb in v['toolboxes'] if tb not in toolboxes]

    # https://openai.github.io/openai-agents-python/ref/model_settings/
    parallel_tool_calls = True if os.getenv('MODEL_PARALLEL_TOOL_CALLS') else False
    model_settings = ModelSettings(
        temperature=os.getenv('MODEL_TEMP', default=0.0),
        tool_choice=('auto' if toolboxes else None),
        parallel_tool_calls=(parallel_tool_calls if toolboxes else None))

    # block tools if requested
    tool_filter = create_static_tool_filter(blocked_tool_names=blocked_tools) if blocked_tools else None

    # fetch mcp params
    mcp_params = mcp_client_params(available_tools.toolboxes, toolboxes)
    for tb, (params, confirms, server_prompt, client_session_timeout) in mcp_params.items():
        server_prompts.append(server_prompt)
        # https://openai.github.io/openai-agents-python/mcp/
        # allowed_tool_names will allow list
        # blocked_tool_names will block list
        if headless:
            # XXX: auto-allow all tools if task is headless by clearing confirms
            confirms = []
        client_session_timeout = client_session_timeout or DEFAULT_MCP_CLIENT_SESSION_TIMEOUT
        server_proc = None
        match params['kind']:
            # since we spawn stdio servers each time we do not expect
            # new tools to appear over time so cache the tools list
            case 'stdio':
                if params.get('reconnecting', False):
                    mcp_server = ReconnectingMCPServerStdio(
                        name=tb,
                        params=params,
                        tool_filter=tool_filter,
                        client_session_timeout_seconds=client_session_timeout,
                        cache_tools_list=True)
                else:
                    mcp_server = MCPServerStdio(
                        name=tb,
                        params=params,
                        tool_filter=tool_filter,
                        client_session_timeout_seconds=client_session_timeout,
                        cache_tools_list=True)
            case 'sse':
                mcp_server = MCPServerSse(
                    name=tb,
                    params=params,
                    tool_filter=tool_filter,
                    client_session_timeout_seconds=client_session_timeout)
            case 'streamable':
                # check if we need to start this server locally as well
                if 'command' in params:
                    def _print_out(line):
                        msg = f"Streamable MCP Server stdout: {line}"
                        logging.info(msg)
                        #print(msg)
                    def _print_err(line):
                        msg = f"Streamable MCP Server stderr: {line}"
                        logging.info(msg)
                        #print(msg)
                    server_proc = StreamableMCPThread(params['command'],
                                                      url=params['url'],
                                                      env=params['env'],
                                                      on_output=_print_out,
                                                      on_error=_print_err)
                mcp_server = MCPServerStreamableHttp(
                    name=tb,
                    params=params,
                    tool_filter=tool_filter,
                    client_session_timeout_seconds=client_session_timeout)
            case _:
                raise ValueError(f"Unsupported MCP transport {params['kind']}")
        # provide namespace and confirmation control through wrapper class
        mcp_servers.append((MCPNamespaceWrap(confirms, mcp_server), server_proc))

    # connect mcp servers
    # https://openai.github.io/openai-agents-python/ref/mcp/server/
    async def mcp_session_task(
            mcp_servers: list,
            connected: asyncio.Event,
            cleanup: asyncio.Event) -> None:
        try:
            # connects/cleanups have to happen in the same task
            # but we also want to use wait_for to set a timeout
            # so we use a dedicated session task to accomplish both
            for s in mcp_servers:
                server, server_proc = s
                logging.debug(f"Connecting mcp server: {server._name}")
                if server_proc is not None:
                    server_proc.start()
                    await server_proc.async_wait_for_connection(poll_interval=0.1)
                await server.connect()
            # signal that we're connected
            connected.set()
            # wait until we're told to clean up
            await cleanup.wait()
            for s in reversed(mcp_servers):
                server, server_proc = s
                try:
                    logging.debug(f"Starting cleanup for mcp server: {server._name}")
                    await server.cleanup()
                    logging.debug(f"Cleaned up mcp server: {server._name}")
                    if server_proc is not None:
                        server_proc.stop()
                        try:
                            await asyncio.to_thread(server_proc.join_and_raise)
                        except Exception as e:
                            print(f"Streamable mcp server process exception: {e}")
                except asyncio.CancelledError:
                    logging.error(f"Timeout on cleanup for mcp server: {server._name}")
                finally:
                    mcp_servers.remove(s)
        except RuntimeError as e:
            logging.error(f"RuntimeError in mcp session task: {e}")
        except asyncio.CancelledError as e:
            logging.error(f"Timeout on main session task: {e}")
            pass
        finally:
            mcp_servers.clear()

    servers_connected = asyncio.Event()
    start_cleanup = asyncio.Event()
    mcp_sessions = asyncio.create_task(
        mcp_session_task(
            mcp_servers,
            servers_connected,
            start_cleanup))

    # wait for the servers to be connected
    await servers_connected.wait()
    logging.debug("All mcp servers are connected!")

    try:

        # any important general guidelines go here
        important_guidelines = [
            "Do not prompt the user with questions.",
            "Run tasks until a final result is available.",
            "Ensure responses are based on the latest information from available tools.",
            "Run tools sequentially, wait until one tool has completed before calling the next.",
        ]

        # create one layer of handoff agents if any additional agents are listed
        # https://openai.github.io/openai-agents-python/handoffs/
        handoffs = []
        for handoff_agent in list(agents.keys())[1:]:
            handoffs.append(TaskAgent(
                # XXX: name has to be descriptive for an effective handoff
                name=compress_name(handoff_agent),
                instructions=prompt_with_handoff_instructions(
                    mcp_system_prompt(
                        agents[handoff_agent]['personality'],
                        agents[handoff_agent]['task'],
                        server_prompts=server_prompts,
                        important_guidelines=important_guidelines)
                ),
                handoffs=[],
                exclude_from_context=exclude_from_context,
                mcp_servers=[s[0] for s in mcp_servers],
                model=model,
                model_settings=model_settings,
                run_hooks=run_hooks,
                agent_hooks=agent_hooks).agent)

        # create the primary task agent
        primary_agent = list(agents.keys())[0]
        system_prompt = mcp_system_prompt(
            agents[primary_agent]['personality'],
            agents[primary_agent]['task'],
            server_prompts=server_prompts,
            important_guidelines=important_guidelines)
        agent0 = TaskAgent(
            name=primary_agent,
            # only add the handoff prompt if we have handoffs defined
            instructions=prompt_with_handoff_instructions(system_prompt) if handoffs else system_prompt,
            handoffs=handoffs,
            exclude_from_context=exclude_from_context,
            mcp_servers=[s[0] for s in mcp_servers],
            model=model,
            model_settings=model_settings,
            run_hooks=run_hooks,
            agent_hooks=agent_hooks)

        try:

            complete = False
            async def _run_streamed():
                max_retry = MAX_API_RETRY
                rate_limit_backoff = RATE_LIMIT_BACKOFF
                while rate_limit_backoff:
                    try:
                        result = agent0.run_streamed(prompt, max_turns=max_turns)
                        # render result events
                        # https://openai.github.io/openai-agents-python/ref/stream_events/
                        # https://openai.github.io/openai-agents-python/ref/run/
                        # https://openai.github.io/openai-agents-python/results/
                        async for event in result.stream_events():
                            if event.type == "raw_response_event" and isinstance(
                                    event.data,
                                    ResponseTextDeltaEvent):
                                await render_model_output(event.data.delta,
                                                    async_task=async_task,
                                                    task_id=task_id)
                        await render_model_output('\n\n',
                                            async_task=async_task,
                                            task_id=task_id)
                        return
                    except APITimeoutError:
                        if not max_retry:
                            logging.error(f"Max retries for APITimeoutError reached")
                            raise
                        max_retry -= 1
                    except RateLimitError:
                        if rate_limit_backoff == MAX_RATE_LIMIT_BACKOFF:
                            raise APITimeoutError(f"Max rate limit backoff reached")
                        if rate_limit_backoff > MAX_RATE_LIMIT_BACKOFF:
                            rate_limit_backoff = MAX_RATE_LIMIT_BACKOFF
                        else:
                            rate_limit_backoff += rate_limit_backoff
                        logging.error(f"Hit rate limit ... holding for {rate_limit_backoff}")
                        await asyncio.sleep(rate_limit_backoff)
            await _run_streamed()
            complete = True

        # raise exceptions up to here for anything that indicates a task failure
        except MaxTurnsExceeded as e:
            await render_model_output(f"** 🤖❗ Max Turns Reached: {e}\n",
                                async_task=async_task,
                                task_id=task_id)
            logging.error(f"Exceeded max_turns: {max_turns}")
        except AgentsException as e:
            await render_model_output(f"** 🤖❗ Agent Exception: {e}\n",
                                async_task=async_task,
                                task_id=task_id)
            logging.error(f"Agent Exception: {e}")
        except BadRequestError as e:
            await render_model_output(f"** 🤖❗ Request Error: {e}\n",
                                async_task=async_task,
                                task_id=task_id)
            logging.error(f"Bad Request: {e}")
        except APITimeoutError as e:
            await render_model_output(f"** 🤖❗ Timeout Error: {e}\n",
                                async_task=async_task,
                                task_id=task_id)
            logging.error(f"Bad Request: {e}")

        if async_task:
            await flush_async_output(task_id)

        return complete

    finally:

        # signal mcp sessions task that it can disconnect our servers
        start_cleanup.set()
        cleanup_attempts_left = len(mcp_servers)
        while cleanup_attempts_left and len(mcp_servers):
            try:
                cleanup_attempts_left -= 1
                await asyncio.wait_for(mcp_sessions, timeout=MCP_CLEANUP_TIMEOUT)
            except asyncio.TimeoutError as e:
                continue
            except Exception as e:
                logging.error(f"Exception in mcp server cleanup task: {e}")


async def main(available_tools: AvailableTools,
               p: str | None, t: str | None, prompt: str | None):
    last_mcp_tool_results = [] # XXX: memleaky

    async def on_tool_end_hook(
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            tool: Tool,
            result: str):
        last_mcp_tool_results.append(result)

    async def on_tool_start_hook(
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            tool: Tool):
        await render_model_output(f"\n** 🤖🛠️ Tool Call: {tool.name}\n")

    async def on_handoff_hook(
            context: RunContextWrapper[TContext],
            agent: Agent[TContext],
            source: Agent[TContext]):
        await render_model_output(f"\n** 🤖🤝 Agent Handoff: {source.name} -> {agent.name}\n")

    if p:
        personality = available_tools.personalities.get(p)
        if personality is None:
            raise ValueError(f"No such personality: {p}")

        await deploy_task_agents(
            available_tools,
            { p:personality },
            prompt,
            run_hooks=TaskRunHooks(
                on_tool_end=on_tool_end_hook,
                on_tool_start=on_tool_start_hook))

    if t:

        taskflow = available_tools.taskflows.get(t)
        if taskflow is None:
            taskflow_list = '\n'.join(available_tools.taskflows.keys())
            await render_model_output(
                f"** 🤖❗ Input Error: No such taskflow: {t}. Available taskflows are:\n{taskflow_list}"
            )
            raise ValueError(f"No such taskflow: {t}")

        await render_model_output(f"** 🤖💪 Running Task Flow: {t}\n")

        # optional global vars available for the taskflow tasks
        global_variables = taskflow.get('globals', {})
        model_config = taskflow.get('model_config', {})
        model_keys = []
        if model_config:
            model_dict = available_tools.model_config.get(model_config, {})
            if not model_dict:
                raise ValueError(f"No such model config: {model_config}")
            model_dict = model_dict.get('models', {})
            if model_dict:
                if not isinstance(model_dict, dict):
                    raise ValueError(f"Models section of the model_config file {model_config} must be a dictionary")
            model_keys = model_dict.keys() 

        for task in taskflow['taskflow']:

            task_body = task['task']

            # reusable taskflow support (they have to be single step taskflows)
            # if uses: is set, swap in the appropriate task_body values from child
            # child values can NOT overwrite existing parent values, so parents
            # can tweak reusable task configurations as they see fit
            uses = task_body.get('uses', '')
            if uses:
                reusable_taskflow = available_tools.taskflows.get(uses)
                if reusable_taskflow is None:
                    raise ValueError(f"No such reusable taskflow: {uses}")
                if len(reusable_taskflow['taskflow']) > 1:
                    raise ValueError("Reusable taskflows can only contain 1 task")
                for k,v in reusable_taskflow['taskflow'][0]['task'].items():
                    if k not in task_body:
                        task_body[k] = v
            model = task_body.get('model', DEFAULT_MODEL)
            if model in model_keys:
                model = model_dict[model]
            # parse our taskflow grammar
            name = task_body.get('name', 'taskflow') # placeholder, not used yet
            description = task_body.get('description', 'taskflow') # placeholder not used yet
            agents = task_body.get('agents', [])
            headless = task_body.get('headless', False)
            blocked_tools = task_body.get('blocked_tools', [])
            run = task_body.get('run', '')
            inputs = task_body.get('inputs', {})
            prompt = task_body.get('user_prompt', '')
            if run and prompt:
                raise ValueError('shell task and prompt task are mutually exclusive!')
            must_complete = task_body.get('must_complete', False)
            max_turns = task_body.get('max_steps', DEFAULT_MAX_TURNS)
            toolboxes_override = task_body.get('toolboxes', [])
            env = task_body.get('env', {})
            repeat_prompt = task_body.get('repeat_prompt', False)
            # this will set Agent 'stop_on_first_tool' tool use behavior, which prevents output back to llm
            exclude_from_context = task_body.get('exclude_from_context', False)
            # this allows you to run repeated prompts concurrently with a limit
            async_task = task_body.get('async', False)
            max_concurrent_tasks = task_body.get('async_limit', 5)

            def preprocess_prompt(prompt: str, tag: str, kv: dict, kv_subkey=None):
                _prompt = prompt
                for full_match in re.findall(r"\{\{\s+" + tag + r"_(?:.*?)\s+\}\}", prompt):
                    _m = re.search(r"\{\{\s+" + tag + r"_(.*?)\s+\}\}", full_match)
                    if _m:
                        key = _m.group(1)
                        if key in kv:
                            _prompt = _prompt.replace(
                                full_match,
                                str(kv.get(key)[kv_subkey]) if kv_subkey else str(kv.get(key)))
                        else:
                            raise KeyError(f"No such prompt key available: {key}")
                return _prompt

            # pre-process the prompt for any prompts
            if prompt:
                prompt = preprocess_prompt(prompt, 'PROMPTS', available_tools.prompts, 'prompt')

            # pre-process the prompt for any inputs
            if prompt and inputs:
                prompt = preprocess_prompt(prompt, 'INPUTS', inputs)

            # pre-process the prompt for any globals
            if prompt and global_variables:
                prompt = preprocess_prompt(prompt, 'GLOBALS', global_variables)

            with TmpEnv(env):
                prompts_to_run = []
                if repeat_prompt:
                    pattern = r"\{\{\s+RESULT_*(.*?|)\s+\}\}"
                    m = re.search(pattern, prompt)
                    # if last mcp tool result is an iterable it becomes available for repeat prompts
                    if not m:
                        logging.critical("Expected templated prompt, aborting!")
                        break
                    try:
                        # if this is json loadable, then it might be an iter, so check for that
                        last_result = json.loads(last_mcp_tool_results.pop())
                        text = last_result.get('text', '')
                        try:
                            iterable_result = json.loads(text)
                        except json.decoder.JSONDecodeError as exc:
                            e = f"Could not json.loads result text: {text}"
                            logging.critical(e)
                            raise ValueError(e) from exc
                        iter(iterable_result)
                    except IndexError:
                        logging.critical("No last mcp tool result available, aborting!")
                        raise
                    except ValueError:
                        logging.critical("Could not json.loads last mcp tool results, aborting!")
                        raise
                    except TypeError:
                        logging.critical("Last mcp tool results are not iterable, aborting!")
                        raise
                    if not iterable_result:
                        await render_model_output("** 🤖❗MCP tool result iterable is empty!\n")
                    else:
                        # we use our own template marker here so prompts are not limited to use {}
                        logging.debug(f"Entering templated prompt loop for results: {iterable_result}")
                        for value in iterable_result:
                            # support RESULT_key -> value swap format as well
                            if isinstance(value, dict) and m.group(1):
                                _prompt = prompt
                                for full_match in re.findall(r"\{\{\s+RESULT_(?:.*?)\s+\}\}", prompt):
                                    _m = re.search(r"\{\{\s+RESULT_(.*?)\s+\}\}", full_match)
                                    if _m and _m.group(1) in value:
                                        _prompt = _prompt.replace(
                                            full_match,
                                            pformat(value.get(_m.group(1))))
                                prompts_to_run.append(_prompt)
                            else:
                                prompts_to_run.append(
                                    prompt.replace(
                                        m.group(0),
                                        pformat(value)))
                else:
                    prompts_to_run.append(prompt)

                async def run_prompts(async_task=False, max_concurrent_tasks=5):

                    # if this is a shell task, execute that and append the results
                    if run:
                        await render_model_output(f"** 🤖🐚 Executing Shell Task\n")
                        # this allows e.g. shell based jq output to become available for repeat prompts
                        try:
                            result = shell_tool_call(run).content[0].model_dump_json()
                            last_mcp_tool_results.append(result)
                            return True
                        except RuntimeError as e:
                            await render_model_output(f"** 🤖❗ Shell Task Exception: {e}\n")
                            logging.error(f"Shell task error: {e}")
                            return False

                    tasks = []
                    task_results = []
                    semaphore = asyncio.Semaphore(max_concurrent_tasks)
                    for prompt in prompts_to_run:
                        # run a task prompt
                        resolved_agents = {}
                        if not agents:
                            # XXX: deprecate the -p parser for taskflows entirely?
                            # XXX: probably just adds unneeded parsing complexity
                            p, _, _, prompt, _ = parse_prompt_args(available_tools, prompt)
                            agents.append(p)
                        for p in agents:
                            personality = available_tools.personalities.get(p)
                            if personality is None:
                                raise ValueError(f"No such personality: {p}")
                            resolved_agents[p] = personality

                        # limit the max concurrent tasks via a semaphore
                        async def _deploy_task_agents(resolved_agents, prompt):
                            async with semaphore:
                                result = await deploy_task_agents(
                                    available_tools,
                                    # pass agents and prompt by assignment, they change in-loop
                                    resolved_agents,
                                    prompt,
                                    async_task=async_task,
                                    toolboxes_override=toolboxes_override,
                                    blocked_tools=blocked_tools,
                                    headless=headless,
                                    exclude_from_context=exclude_from_context,
                                    max_turns=max_turns,
                                    run_hooks=TaskRunHooks(
                                        on_tool_end=on_tool_end_hook,
                                        on_tool_start=on_tool_start_hook),
                                    model = model,
                                    agent_hooks=TaskAgentHooks(
                                        on_handoff=on_handoff_hook))
                            return result

                        task_coroutine = _deploy_task_agents(resolved_agents, prompt)

                        if not async_task:
                            # wait for the task
                            result = await task_coroutine
                            task_results.append(result)
                        else:
                            # stack the task
                            tasks.append(task_coroutine)

                    if async_task:
                        # gather results
                        task_results = await asyncio.gather(*tasks, return_exceptions=True)

                    complete = True
                    # if any prompt in a must_complete task is not complete the entire task is incomplete
                    for result in task_results:
                        if isinstance(result, Exception):
                            logging.error(f"Caught exception in Gather: {result}")
                            result = False
                        complete = result and complete
                    return complete

                # an async tasks runs prompts concurrently
                task_complete = await run_prompts(
                    async_task=async_task,
                    max_concurrent_tasks=max_concurrent_tasks)

                if must_complete and not task_complete:
                    logging.critical("Required task not completed ... aborting!")
                    await render_model_output("🤖💥 *Required task not completed ...\n")
                    break

if __name__ == '__main__':
    cwd = pathlib.Path.cwd()
    available_tools = AvailableTools(
        YamlParser(cwd).get_yaml_dict((cwd/'personalities').rglob('*')) |
        YamlParser(cwd).get_yaml_dict((cwd/'taskflows').rglob('*')) |
        YamlParser(cwd).get_yaml_dict((cwd/'prompts').rglob('*')) |
        YamlParser(cwd).get_yaml_dict((cwd/'toolboxes').rglob('*')) |
        YamlParser(cwd).get_yaml_dict((cwd/'configs').rglob('*')))

    p, t, l, user_prompt, help_msg = parse_prompt_args(available_tools)

    if l:
        tool_models = list_tool_call_models(os.getenv('COPILOT_TOKEN'))
        for model in tool_models:
            print(model)
        sys.exit(0)

    if p is None and t is None:
        print(help_msg)
        sys.exit(1)

    asyncio.run(main(available_tools, p, t, user_prompt), debug=True)
