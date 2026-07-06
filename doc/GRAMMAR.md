# Taskflows

Taskflows are YAML lists of tasks. They are specified by the `filetype` `taskflow`.

Example:

```yaml
taskflow:
  - task:
    ...
  - task:
    ...
```

## Tasks

Tasks define, at minimum, a list of Agents to use and a User Prompt.

Example:

```yaml
  - task:
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        This is a user prompt.
```

Note: The exception to this rule are `run` shell tasks.

### Agents

`agents` defines the system prompt to be used for the task. It contains a list of files of type `personality`.

For example, to use the `personality` defined in the following:

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: personality

personality: |
  You are a helpful assistant.
  
task: |
  Your primary task is to use available tools to complete user defined tasks.

  Always use available tools to complete your tasks. If the tools you require
  to complete a task are not available, politely decline the task.
  
toolboxes:
  - ...
```

The task should include the personality YAML file in its list of `agents`:

```yaml
  - task:
      agents:
        - seclab_taskflow_agent.personalities.assistant
  ...
```

Task agent lists can define one (primary) or more (handoff) agents.

Example:

```yaml
  - task:
      agents:
        - primary_agent
        - handoff_agent1
        - ...
        - handoff_agentN
      user_prompt: |
        ...
```

### Model

Tasks can optionally specify which Model to use on the configured inference endpoint:

```yaml
  - task:
      model: gpt-4.1
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        This is a user prompt.
```

Note that model identifiers may differ between OpenAI compatible endpoint providers, make sure you change your model identifier accordingly when switching providers. If not specified, a default LLM model (such as `gpt-4.1`) is used.

Parameters to the model can also be specified in the task using the `model_settings` section:

```yaml
    model: gpt-5-mini
    model_settings:
      temperature: 1
      reasoning:
        effort: high
```

If `model_settings` is absent, then the model parameters will fall back to either the default or the ones supplied in a `model_config`. However, any parameters supplied in the task will override those that are set in the `model_config`.

### Multiple Models (multi-model tasks)

A task can be run against several models at once using the `models` field. Each
model runs the task in parallel and its output is streamed as its own labelled
block, which makes it easy to compare how different models respond to the same
prompt (for example when evaluating an audit prompt across model families).

The simplest form is a list of logical model names (resolved through the
`model_config` exactly like the singular `model` field):

```yaml
  - task:
      models: [gpt_default, claude_native, gpt_responses]
      agents:
        - seclab_taskflow_agent.personalities.c_auditer
      user_prompt: |
        Audit this function for memory safety issues.
```

Each entry may also be a map with its own `model_settings`, so different models
can use different parameters in the same task:

```yaml
  - task:
      models:
        - model: gpt_default
          model_settings:
            temperature: 0.2
        - model: claude_native
          model_settings:
            reasoning:
              effort: high
      agents:
        - seclab_taskflow_agent.personalities.c_auditer
      user_prompt: |
        Audit this function for memory safety issues.
```

Notes and semantics:

- `model` (singular) and `models` (plural) are mutually exclusive. `model: x`
  is exactly equivalent to `models: [x]`; existing single-model taskflows are
  unaffected.
- Per-entry `model_settings` support the same engine keys as `model_config`
  (`api_type`, `endpoint`, `token`, `backend`), so different models may run on
  different backends within one task.
- `completion` controls when the task counts as complete across its fan-out
  branches: `all` (default, every branch must succeed) or `any` (one branch
  succeeding is enough). This is what `must_complete` checks against.
- `model_concurrency` caps how many branches run at once for a multi-model
  task (default `0` runs all models in parallel):

```yaml
  - task:
      models: [m1, m2, m3, m4]
      model_concurrency: 2
      completion: any
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        ...
```

- Multi-model output is buffered per model and flushed as a labelled block when
  each model finishes, so streams from different models do not interleave.
- `models` can be combined with `repeat_prompt`: the task runs the cross
  product of items x models concurrently, bounded by `model_concurrency`
  (default: all branches at once). Each branch is streamed as its own block
  labelled `<model> [item <n>]`.
- Multi-model tool results are not threaded into the implicit last-tool-result
  channel that the next task's `repeat_prompt` reads (that stays deterministic).
  To consume multi-model results downstream, give the task an `id`: each
  branch's final result is aggregated into `outputs.<id>` as a list of records
  `{"model": ..., "item": ..., "result": ...}` (see "Typed named outputs").
- The inline `outputs` schema is applied per branch on multi-model tasks: each
  branch's result is validated against the schema and stored as the
  `result` of its fan-in record. A branch whose result violates the schema is
  treated as a failed branch, which the `completion` policy (`all`/`any`) then
  reduces like any other branch failure (see "Typed named outputs").

### Completion Requirement

Tasks can be marked as requiring completion, if a required task fails, the taskflow will abort. This defaults to false.

Example:

```yaml
  - task:
      must_complete: true
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        ...
```

### Conditional execution (`if`)

A task can be gated with a GitHub-Actions-style `if` condition: a Jinja
expression evaluated against the template context (`globals`, `inputs`, and
prior tasks' `outputs`). When it evaluates falsy the task is skipped (recorded
as skipped and not run); otherwise it runs normally.

```yaml
  - task:
      id: audit
      agents: [seclab_taskflow_agent.personalities.c_auditer]
      user_prompt: |
        Audit this code and report findings as JSON: {"findings": [...]}
      outputs:
        type: object
        properties:
          findings:
            type: array
        required: [findings]
  - task:
      # only remediate when the audit actually found something
      if: "outputs.audit.findings | length > 0"
      agents: [seclab_taskflow_agent.personalities.assistant]
      user_prompt: |
        Propose fixes for: {{ outputs.audit.findings }}
```

Notes:

- The expression may be written bare (`globals.mode == 'deep'`) or wrapped in
  `{{ ... }}`. Standard truthiness applies (empty list/string/`0`/`false` are
  falsy).
- Referencing a name that does not exist (for example an output from a task that
  has not run) is treated as falsy, so the task is skipped rather than failing,
  matching GitHub Actions semantics. To be explicit you can still guard with
  `is defined`, e.g. `if: "outputs.audit is defined and outputs.audit.findings"`.
- `if` composes with everything else: a skipped task does not run its agents,
  fan out over `models`, or capture outputs.

### Conditionals and loops inside a prompt

Because prompts are rendered with Jinja2, you can use `{% if %}` / `{% for %}`
directly inside a `user_prompt`:

```yaml
  - task:
      agents: [seclab_taskflow_agent.personalities.c_auditer]
      user_prompt: |
        {% if globals.mode == 'deep' %}
        Perform a DEEP audit of {{ globals.target }}.
        {% else %}
        Do a quick scan of {{ globals.target }}.
        {% endif %}
        {% for area in globals.focus %}
        - pay attention to {{ area }}
        {% endfor %}
```

Unlike the task-level `if` condition (which treats undefined names as falsy and
skips the task), undefined variables inside a prompt raise, to catch typos. Use
`is defined` or the `default` filter for optional data:
`{{ globals.note | default('') }}`.

### Running templated tasks in a loop

Often we may want to iterate through the same tasks with different inputs. For example, we may want to fetch all the functions from a code base and then analyze each of the functions. This can be done using two consecutive tasks and with the help of the `repeat_prompt` field. 

```yaml
  - task:
    agents:
      - seclab_taskflow_agent.personalities.assistant
    user_prompt: |
      Fetch all the functions in the code base and create a list with entries of the form {'name' : <function_name>, 'body' : <function_body>}
  - task:
    repeat_prompt: true
    agents:
      - seclab_taskflow_agent.personalities.c_auditer
    user_prompt: |
      The function has name {{ result.name }} and body {{ result.body }} analyze the function.
```

In the above, the first task fetches functions in the code base and creates a json list object, with each entry having a `name` and `body` field. In the next task, `repeat_prompt` is set to true, meaning that a task is created for each individual object in the list and the object fields are referenced in the templated prompt using `{{ result.fieldname }}`. In other words, `{{ result.name }}` in the prompt is replaced with the value of the `name` field of the object etc. For example, if the list of functions fetched from the first task is:

```javascript
[{'name' : foo, 'body' : foo(){return 1;}}, {'name' : bar, 'body' : bar(a) {return a + 1;}}]
```

Then the tasks created will have their prompts replaced by:

```yaml
      The function has name foo and body foo(){return 1;} analyze the function.
```

etc. 

Note that when using `repeat_prompt`, the last tool call result of the previous task is used as the iterable. It is recommended to keep the task that creates the iterable short and simple (e.g. just make one tool call to fetch a list of results) to avoid wrong results being passed to the repeat prompt.

The iterable can also contain a list of primitives like string or number, in which case, the template `{{ result }}` can be used in the `repeat_prompt` prompt to parse the results instead:

```yaml
  - task:
      max_steps: 5
      must_complete: true
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        Store the json array [1, 2, 3] in memory under the
        `test_repeat_prompt` key as a json object, then retrieve
        the contents of the `test_repeat_prompt` key from memory
        ...
  - task:
      # if the last mcp tool result is iterable
      # repeat_prompt can iter those results
      must_complete: true
      repeat_prompt: true
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        What is the integer value of {{ result }}?
```

Repeat prompt can be run in parallel by setting the `async` field to `true`:

```yaml
  - task:
    repeat_prompt: true
    async: true
    agents:
      - seclab_taskflow_agent.personalities.c_auditer
    user_prompt: |
      The function has name {{ result.name }} and body {{ result.body }} analyze the function.
```

An optional limit can be set to limit the number of asynchronous tasks via `async_limit`. If not set, the default value (5) is used.

```yaml
  - task:
    repeat_prompt: true
    async: true
    async_limit: 3
    agents:
      - seclab_taskflow_agent.personalities.c_auditer
    user_prompt: |
      The function has name {{ result.name }} and body {{ result.body }} analyze the function.
```

Both `async` and `async_limit` have no effect when used outside of a `repeat_prompt`.

At the moment, we do not support nested `repeat_prompt`. So the following is not allowed:

```yaml
  - task:
    repeat_prompt: true
    agents:
      - seclab_taskflow_agent.personalities.c_auditer
    user_prompt: |
      The function has name {{ result.name }} and body {{ result.body }} analyze the function.
  - task:
    repeat_prompt: true
    ...
```

#### Shell Tasks

Tasks can be entirely shell based through the run directive. This simply runs a shell command and pass the result directly to the next task. It can be used for creating iterable results for `repeat_prompt`.

For example:

```yaml
  - task:
      must_complete: true
      run: |
        echo '["apple", "banana", "orange"]'
  - task:
      repeat_prompt: true
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        What kind of fruit is {{ result }}?
```

The string `["apple", "banana", "orange"]` is then passed directly to the next task.

This allows you to e.g. pass in json iterable outputs from shellscripts into a prompt task.

Use shell tasks when you want to iterate on results that don't need to be generated via a tool call.

#### Context Exclusion

Often when creating iterable results for a `repeat_prompt`, a large iterable is created and we do not want it to be passed to the LLM model because it can easily exceed the token limit. In this case, tasks can specify that their tool results and output should be available at the Agent level but not included in the Model context using the `exclude_from_context` field.

Example:

```yaml
  - task:
      exclude_from_context: true
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        List all the files in the codeql database `some/codeql/db`.
      toolboxes:
        - seclab_taskflow_agent.toolboxes.codeql
```

### Typed named outputs

By default, data flows between tasks implicitly: `repeat_prompt` consumes the
*last tool result* of the previous task. That is positional (whichever tool
fired last) and untyped. A task can instead publish a **named, typed output**
that later tasks consume by name.

Three fields drive this:

- `id` names a task, exposing its output to later tasks as `outputs.<id>`. The
  shape depends on whether the task fans out:
  - A plain task (single model, no `repeat_prompt`) publishes its single
    produced value (its final tool result).
  - A task that fans out (`repeat_prompt`, multiple `models`, or their cross
    product) publishes a per-branch fan-in list of
    `{"model": <label>, "item": <index>, "result": <value>}` records, one per
    branch. This is uniform across the item axis and the model axis, so a
    single-model `repeat_prompt` and a multi-model task capture the same way.
- `outputs` declares an inline JSON Schema (Draft 2020-12). When present, the
  task's value is validated against it before being stored. Validation is strict
  and does not coerce, so a value whose types do not match the contract is a
  failure. On a fan-out task the schema is applied to each branch's `result`
  (a violation is a failed branch under the `completion` policy); on a plain
  task it is applied to the single value (a violation is a hard failure). A
  malformed schema is rejected when the taskflow is loaded, before any model
  calls are made.
- `over` is an explicit iterable selector for `repeat_prompt`: a Jinja
  expression evaluated against the template context (so it yields a real list,
  not a re-parsed string).

Example: one task produces a typed list of functions, the next analyses each.

```yaml
  - task:
      id: list_functions
      agents: [seclab_taskflow_agent.personalities.assistant]
      user_prompt: |
        List all functions as JSON: {"functions": [{"name": ..., "body": ...}]}
      outputs:
        type: object
        properties:
          functions:
            type: array
            items:
              type: object
              properties:
                name: {type: string}
                body: {type: string}
              required: [name, body]
        required: [functions]
  - task:
      repeat_prompt: true
      over: "outputs.list_functions.functions"
      agents: [seclab_taskflow_agent.personalities.c_auditer]
      user_prompt: |
        Analyze function {{ result.name }}:
        {{ result.body }}
```

The `outputs` schema is a standard JSON Schema (Draft 2020-12), authored inline
in YAML, so the full vocabulary is available:

- Types via `type` (`object`, `array`, `string`, `integer`, `number`,
  `boolean`, `null`), with `properties`/`required` for objects and `items` for
  arrays.
- `enum`/`const` for fixed value sets, and constraints such as `minimum`,
  `maximum`, `minLength`, and `pattern`.
- Objects with dynamic keys via `additionalProperties`, and strictness via
  `additionalProperties: false`.
- Unions (`anyOf`/`oneOf`), and `$ref`/`$defs` to reuse a shape across fields.

Because validation is strict (no coercion), have the task emit JSON whose types
already match, e.g. the integer `7` rather than the string `"7"`.

Notes:

- `outputs.<id>` is a template namespace alongside `globals`, `inputs`, and
  the per-iteration `result`. Keys named after dict methods (`items`, `keys`,
  `values`, ...) resolve to your data, not the method.
- Without `id`/`outputs`/`over`, the implicit last-tool-result `repeat_prompt`
  behaviour is unchanged.
- Implicit carry-over (the next task's `repeat_prompt` reading the previous
  task's last tool result) is fed by single-model tasks only. A multi-model
  task does not feed it: there is no single "last" result across models, so a
  downstream task must consume its output by name via `id`/`over`.

### Toolboxes / MCP Servers

Toolboxes are MCP server configurations. They can be defined at the Agent level or overridden at the task level. These MCP servers are started and made available to the Agents in the Agents list during a Task. The `toolboxes` field should contain a list of files for the `toolboxes` that are available for the task:

```yaml
  - task:
      ...
      toolboxes:
        - seclab_taskflow_agent.toolboxes.codeql
```

If no `toolboxes` are specified, then the `toolboxes` defined in the `personality` of the `agent` are used:

```yaml
   - task:
      agents:
        - seclab_taskflow_agent.personalities.c_auditer
      user_prompt: |
        List all the files in the codeql database `some/codeql/db`.      
   - task:
```

In the above `task`, as no `toolboxes` is specified, the `toolboxes` defined in the `personality` of `seclab_taskflow_agent.personalities.c_auditer` is used.

Note that when `toolboxes` is defined for a task, it *overwrites* the `toolboxes` that are available. For example, in the following `task`:

```yaml
   - task:
      agents:
        - seclab_taskflow_agent.personalities.c_auditer
      user_prompt: |
        List all the files in the codeql database `some/codeql/db`.      
      toolboxes:
        - seclab_taskflow_agent.toolboxes.echo

```

For this task, the `agent` `seclab_taskflow_agent.personalities.c_auditer` will have access to the `seclab_taskflow_agent.toolboxes.echo` tool.

### Headless Runs

MCP server configurations can request confirmations for tool calls. These confirmations are prompted on the terminal. If you want to allow all tool calls by default for headless use, you can set a task to run headless.

Example:

```yaml
  - task:
      headless: true
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        Clear the memory cache.
      toolboxes:
        - memcache
```

### Environment Variables

Tasks can be configured to set temporary os environment variables available during the task. This is primarily used to pass through configuration options to toolboxes (mcp servers).

Example:

```yaml
  - task:
      headless: true
      agents:
        - seclab_taskflow_agent.personalities.assistant
      user_prompt: |
        Store `hello` in the memory key `world`.
      toolboxes:
        - seclab_taskflow_agent.toolboxes.memcache
      env:
        MEMCACHE_STATE_DIR: "example_taskflow/"
        MEMCACHE_BACKEND: "dictionary_file"
```

### Globals

Taskflows can define toplevel global variables available to every task.

Example:

```yaml
globals:
  fruit: bananas
taskflow:
  - task:
      agents:
        - examples.personalities.fruit_expert
      user_prompt: |
        Tell me more about {{ globals.fruit }}.
```

Global variables can also be set or overridden from the command line using the `-g` or `--global` flag:

```sh
hatch run main -t examples.taskflows.example_globals -g fruit=apples
```

Multiple global variables can be set by repeating the flag:

```sh
hatch run main -t examples.taskflows.example_globals -g fruit=apples -g color=red
```

Command line globals override any globals defined in the taskflow YAML file, allowing you to reuse taskflows with different parameter values without editing the files.

### Reusable Tasks

Tasks can reuse single step taskflows and optionally override any of its configurations. This is done by setting a `uses` field with a link to the single step taskflow YAML file as its value.

Example:

```yaml
  - task:
      uses: examples.taskflows.single_step_taskflow
      model: gpt-4o
```

In this case, the prompt and settings of `single_step_taskflow` is used. However, the `model` parameter is overwritten by `gpt-4o`. For example, if `single_step_taskflow` looks like this:

```yaml
taskflow:
  - task:
      agents:
        - some_agent
      model:
        gpt-4.1
      user_prompt: |
        some actions
      toolboxes:
        - some_toolboxes
```

Then the `task` that uses it effectively becomes:
```yaml
  - task:
      agents:
        - some_agent
      model:
        gpt-4o
      user_prompt: |
        some actions
      toolboxes:
        - some_toolboxes
```

Any `taskflow` that contains only a single step can be used as a reusable taskflow.

A reusable taskflow can also have a templated prompt that takes inputs from its user. This is specified with the `inputs` field from the user.

```yaml
  - task:
      uses: examples.taskflows.single_step_taskflow
      inputs:
        fruit: apples
```

```yaml
  - task:
      agents:
        - examples.personalities.fruit_expert
      user_prompt: |
        Tell me more about {{ inputs.fruit }}.
```

In this case, the template parameter `{{ inputs.fruit }}` is replaced by the value of `fruit` from the `inputs` of the user, which is apples in this case:

```yaml
  - task:
      agents:
        - examples.personalities.fruit_expert
      user_prompt: |
        Tell me more about apples.
```

### Reusable Prompts

Reusable prompts are defined in files of `filetype` `prompts`. These are like macros that get included using Jinja2's `{% include %}` directive.

Tasks can incorporate reusable prompts using the include directive. For example:

Example:

```yaml
  - task:
      agents:
        - examples.personalities.fruit_expert
      user_prompt: |
        Tell me more about apples.

        {% include 'examples.prompts.example_prompt' %}
```
and `examples.prompts.example_prompt` is the following:

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: prompt

prompt: |
  Tell me more about bananas as well.
```

Then the actual task becomes:

```yaml
  - task:
      agents:
        - examples.personalities.fruit_expert
      user_prompt: |
        Tell me more about apples.

        Tell me more about bananas as well.
```

### Model config

LLM models can be configured in a taskflow by setting the `model_config` field to a file of type `model_config`:

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: taskflow

model_config: examples.model_configs.model_config
```

The variables defined in the `model_config` file can then be used throughout the taskflow, e.g.

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config
models:
  gpt_latest: gpt-5
```

When `gpt_latest` is used in the taskflow to specify a model, the value `gpt-5` is used:

```yaml
  - task:
      model: gpt_latest
      must_complete: false
      agents:
        - seclab_taskflow_agent.personalities.c_auditer
      user_prompt: |

```

This provides an easy way to update model versions in a taskflow.

#### Per-model settings

A `model_config` file can include per-model settings via `model_settings` and a
global `api_type` that applies to all models unless overridden:

```yaml
seclab-taskflow-agent:
  version: "1.0"
  filetype: model_config
api_type: chat_completions        # default for all models
models:
  gpt_default: gpt-4.1
  gpt_responses: gpt-5.1
  claude_native: claude-opus-4.7
model_settings:
  gpt_default:
    temperature: 0.7
  gpt_responses:
    api_type: responses           # use the Responses API for this model
    endpoint: https://api.githubcopilot.com
    token: CAPI_TOKEN             # env var name containing the API key
    temperature: 0.5
  claude_native:
    api_type: messages            # use the Anthropic Messages API
    backend: anthropic_sdk
    reasoning:
      effort: high
```

The following keys in `model_settings` are handled by the engine and are not
passed to the underlying model provider:

| Key | Description | Default |
|-----|-------------|---------|
| `api_type` | `"chat_completions"`, `"responses"`, or `"messages"` | Inherited from top-level `api_type`, or `"chat_completions"` |
| `backend` | SDK adapter: `"openai_agents"`, `"copilot_sdk"`, or `"anthropic_sdk"` | Inherited from top-level `backend`, or `"openai_agents"` |
| `endpoint` | API base URL for this model | The global `AI_API_ENDPOINT` env var |
| `token` | Name of an environment variable containing the API key | Uses `AI_API_TOKEN` / `COPILOT_TOKEN` |

All other keys (e.g. `temperature`, `top_p`, `reasoning`) are forwarded to the selected SDK backend. Each backend decides what to do with each key: `openai_agents` accepts the standard OpenAI parameter set; `anthropic_sdk` forwards a curated subset (currently `temperature`, `top_p`, `reasoning`, `max_tokens`, `stream_thinking`, `prompt_caching`) and silently ignores keys outside that set; `copilot_sdk` consumes the keys its SDK exposes (e.g. `reasoning_effort`) and **rejects** unsupported keys at validate time with `BackendCapabilityError` (currently `temperature` and `parallel_tool_calls`) rather than silently dropping them. Consult the backend-specific docs if in doubt.
