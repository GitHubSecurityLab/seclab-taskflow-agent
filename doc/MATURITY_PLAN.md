# Taskflow Grammar Maturity Plan

Status: Draft for iteration. M1 (multi-model tasks), M2 (typed named object
passing on the neutral result foundation), M0 (offline linter, schema export,
corpus-validate gate, OutputRouter isolation), M3 (multi-model x repeat_prompt
cross product + fan-in), and GitHub-Actions-style conditional task execution
(`if`) implemented on branch `anticomputer/grammar-maturity`.
Owner: @anticomputer
Target: a series of backwards-compatible PRs that mature the taskflow grammar
into an enterprise-ready declarative agent workflow language.

This document is a living design plan. It starts from a critical review of the
grammar as it exists today, proposes concrete additive grammar changes, and
sequences them into shippable milestones. Nothing here is final; the intent is
to give us a shared artifact to argue over before we write engine code.

---

## 1. Goals and non-goals

### Goals

- Ship first-class multi-model execution: run one task against N model configs
  in parallel with per-model output streams.
- Replace the naive tool-result passing between tasks with typed, validated,
  named object passing (pydantic-ai-style structured outputs).
- Harden the grammar for enterprise use: strictness, schema tooling, run
  reproducibility, observability, and a real versioning/deprecation policy.
- Preserve backwards compatibility for every existing taskflow, personality,
  toolbox, model_config, and prompt document wherever it is technically
  possible. Where it is not, gate behind an explicit opt-in and a grammar
  version bump.

### Non-goals (for this plan)

- Rewriting the backend adapter layer. The `sdk/` abstraction
  (`AgentSpec` / `AgentBackend` in `src/seclab_taskflow_agent/sdk/base.py`) is
  already a clean seam and we build on it rather than replace it.
- Changing the CLI surface beyond additive flags.
- Introducing a new execution runtime (we keep asyncio + the existing runner
  loop).

---

## 2. Critical review of the current grammar

The grammar is defined by Pydantic models in
`src/seclab_taskflow_agent/models.py` and executed by
`src/seclab_taskflow_agent/runner.py`. Documented in `doc/GRAMMAR.md`.

Overall the design is good: a GitHub-Actions-flavored YAML, clean document
types, a backend adapter seam, session checkpoint/resume, and Jinja2
templating. The gaps below are what stand between "good" and "enterprise
ready".

### 2.1 Single-model tasks

`TaskDefinition.model` is a single `str` (`models.py`). `_resolve_task_model()`
in `runner.py` collapses to exactly one `(model_id, settings, api_type,
endpoint, token, backend)` tuple. To A/B two models on the same task today you
must copy the task. There is no fan-out over models.

The building blocks for fan-out already exist, though: `repeat_prompt async`
fans out over an iterable using an `asyncio.Semaphore` + `asyncio.gather` in
`run_prompts()` (`runner.py`), and per-stream output is buffered by
`render_model_output()` / `flush_async_output()` keyed on `task_id`
(`render_utils.py`). Multi-model is the same fan-out shape with a different
iteration axis.

### 2.2 Object passing between tasks is naive and brittle

This is the sharpest enterprise gap. Cross-task data flow rides entirely on
`last_mcp_tool_results: list[str]` in `runner.py`, populated by the
`on_tool_end` hook. `repeat_prompt` consumes it like this
(`_build_prompts_to_run`):

1. Take only `last_mcp_tool_results[-1]` (the last tool call, whatever it was).
2. `json.loads()` it to get an envelope `{"text": ...}`.
3. `json.loads()` the `text` field again to get an iterable.
4. `iter()` it and render one prompt per element.

Problems:

- Positional and implicit: the "output" of a task is whatever tool happened to
  fire last. `doc/GRAMMAR.md` even warns users to "keep the task that creates
  the iterable short and simple" to avoid picking up the wrong result. That
  warning is a design smell.
- Untyped and unvalidated: two `json.loads` hops over an opaque string. A
  malformed or non-JSON result fails deep in the runner with a generic
  `ValueError`. There is no schema, no field validation, no typed access.
- No names: a task cannot say "produce `functions: list[Function]`" and a
  later task cannot say "consume `functions`". Everything is `{{ result }}`.
- No fan-in: you cannot collect the outputs of a fanned-out task and pass the
  aggregate to the next task. Only `[-1]` survives.
- Shell tasks bolt onto the same channel by pushing
  `shell_tool_call(run).content[0].model_dump_json()`, reinforcing the
  stringly-typed contract.

The user ask ("closer adoption of things like pydantic-ai to pass objects
between tasks") maps directly onto fixing this: give tasks typed, named,
validated outputs.

### 2.3 Grammar is permissive to a fault

Every document model sets `model_config = ConfigDict(extra="allow")`
(`models.py`). That means a typo like `user_promt:` or `must_complte: true` is
silently accepted and silently ignored. For a config language that drives
expensive, long-running agent runs, silent misconfiguration is an
enterprise-grade footgun. There is no strict mode and no way to lint a taskflow
before spending tokens on it.

### 2.4 No schema tooling for authors

There is no exported JSON Schema. Authors get no editor autocomplete, no
in-IDE validation, and no CI-time "is this taskflow even valid" check short of
running it. The Pydantic models are the de facto schema but they are not
published in a consumable form.

### 2.5 Thin observability and no run manifest

Output is streamed to stdout and mirrored into a render log
(`render_utils.py`). Sessions persist task booleans and raw tool-result strings
(`session.py`). There is no structured, machine-readable record of a run:
resolved model IDs, per-model settings, backend, token/cost accounting,
timings, artifacts, or exit status per task/model. Enterprise users need a run
manifest to audit, compare, and reproduce runs, especially once one task can
fan out across many models.

### 2.6 Versioning is a hard gate with no evolution story

`TaskflowHeader` only accepts `version: "1.0"` and rejects everything else
(`SUPPORTED_VERSION` in `models.py`). There is no capability negotiation, no
deprecation window, and no way to introduce a breaking grammar change without
breaking every file at once. An enterprise grammar needs a documented
version/feature-gate policy.

### 2.7 Shared mutable run state limits safe concurrency

`last_mcp_tool_results` (a list) and `async_output` (a module-global dict in
`render_utils.py`) are shared process-wide. `repeat_prompt async` already fans
out concurrently; adding a second concurrency axis (models) over the same
shared state increases the chance of interleaving bugs. We want per-run,
per-branch isolation for result capture and output routing.

### 2.8 Smaller sharp edges

- `repeat_prompt` cannot nest (documented limitation in `doc/GRAMMAR.md`).
- Tasks return a bare `bool`; there is no structured per-task result surfaced
  to the caller or the session beyond success/failure.
- `async` is aliased to `async_task` because it is a Python keyword; fine, but
  worth documenting as a grammar reserved word.

---

## 3. Design principles

1. Additive-first. New capabilities are new optional fields. Existing files
   parse and run unchanged.
2. Singular/plural coexistence. Where we pluralize (`model` -> `models`), the
   singular stays valid and is defined as sugar for the one-element plural.
3. Opt-in strictness. Strict validation and typed contracts are available and
   recommended, but not forced onto legacy files.
4. One concurrency engine. Multi-model fan-out reuses the existing
   semaphore/gather + per-`task_id` output buffering machinery rather than
   inventing a parallel path.
5. Typed at the edges. Object passing between tasks becomes typed and named,
   but untyped `{{ result }}` passing keeps working.
6. Version-gated breakage. Anything that cannot be additive is gated behind a
   grammar version bump plus a deprecation window.

---

## 4. Headline feature A: multi-model tasks

### 4.1 Author-facing grammar

Add an optional `models` field to a task. It accepts a list of logical model
names (the same names used in `model:` today, resolved through `model_config`)
or inline per-model setting maps.

Simple form (list of logical model names):

```yaml
  - task:
      models: [gpt_default, claude_native, gpt_responses]
      agents:
        - seclab_taskflow_agent.personalities.c_auditer
      user_prompt: |
        Audit this function for memory safety issues.
```

Rich form (per-entry overrides, still resolved against model_config):

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

Backwards compatibility: `model:` (singular) is unchanged. `models:` is
optional. `model:` is exactly equivalent to `models: [<that one model>]`.
Specifying both is a validation error with a clear message.

### 4.2 Semantics

- A task with `models: [m1, ..., mN]` executes its prompt(s) N times, once per
  model, concurrently, each through the model's resolved backend
  (`_resolve_task_model` runs per entry, so mixed backends already work).
- Concurrency is bounded by a new optional `model_concurrency` (default: run
  all models in parallel, capped by a sane default such as the number of
  models, with an env override). This is orthogonal to `async_limit`, which
  bounds `repeat_prompt` fan-out.
- Interaction with `repeat_prompt`: the cross product (models x iterable
  elements) is the natural semantics. To keep the first cut tractable and
  safe, phase 1 supports `models` on non-`repeat_prompt` tasks and on
  `repeat_prompt` tasks with an explicit, documented cross-product bounded by
  `model_concurrency * async_limit`. See open questions (section 9).
- `must_complete` semantics for a multi-model task need a policy: does the task
  "complete" if all models succeed, any model succeeds, or a quorum? Proposed
  default: all models must succeed (strict), with an optional
  `completion: any | all | quorum(k)` field. Defaults preserve today's
  single-model behavior (one model, all == any).

### 4.3 Per-model output streams

Reuse the existing per-`task_id` buffering in `render_utils.py`. Each model run
gets its own stream identity (a `(task_id, model_label)` pair). Two rendering
modes:

- Interleaved (default for a small number of models): stream live, each line
  prefixed with a short model label so the user can follow N streams at once.
- Buffered-then-flushed (default for larger N, mirrors `repeat_prompt async`):
  each model's output is buffered and flushed as a labeled block when it
  finishes, via a generalization of `flush_async_output()`.

Implementation direction: introduce a small `OutputRouter` abstraction that
owns the `async_output` dict (removing the module-global) and knows how to
label, buffer, and flush per-branch streams. Both `repeat_prompt async` and
multi-model route through it. This also resolves section 2.7 (shared mutable
output state).

### 4.4 Engine changes (sketch)

- `models.py`: add `models: list[str | ModelEntry]`, `model_concurrency: int`,
  and `completion` to `TaskDefinition`; add a `ModelEntry` submodel; add a
  `model_validator` enforcing the `model` xor `models` rule and normalizing
  `model` into a one-element `models`.
- `runner.py`: factor the current per-model resolution
  (`_resolve_task_model`) into a loop that yields one resolved model spec per
  entry; extend `run_prompts()` to fan out over `(model_spec, prompt)` pairs
  using the existing semaphore/gather pattern; thread a stream label through to
  `deploy_task_agents` and into `render_model_output`.
- `render_utils.py`: extract `OutputRouter`; keep the free functions as thin
  wrappers for backwards compatibility.
- `session.py`: record per-model results (section 6.2) instead of a single
  bool.

### 4.5 Testing

- Unit: model validation for `model`/`models` equivalence and the both-set
  error; normalization of singular to plural.
- Runner: a fake backend that records which `(model, prompt)` pairs it was
  asked to run; assert the full fan-out set and concurrency bound.
- Output: assert labeled buffering/flush ordering with a deterministic fake
  stream.
- No live model calls in unit tests; mirror the existing
  `test_sdk_*_adapter.py` fakes and `test_stream.py` patterns.

---

## 5. Headline feature B: typed object passing between tasks

### 5.1 Problem restatement

Section 2.2. Data flow is a stringly-typed, positional, single-slot channel.
We want named, typed, validated outputs that later tasks consume by name.

### 5.2 Proposed model: named, typed task outputs

Introduce an explicit output contract on a task and a named results namespace
in the template context.

```yaml
  - task:
      id: list_functions
      agents: [seclab_taskflow_agent.personalities.assistant]
      user_prompt: |
        List all functions as JSON: [{"name": ..., "body": ...}, ...]
      outputs:
        functions:
          type: list
          items:
            name: str
            body: str

  - task:
      repeat_prompt: true
      over: "{{ outputs.list_functions.functions }}"
      agents: [seclab_taskflow_agent.personalities.c_auditer]
      user_prompt: |
        Analyze function {{ result.name }}:
        {{ result.body }}
```

Key ideas:

- `id` names a task so its outputs are addressable as
  `outputs.<id>.<field>` in later prompts (replacing the implicit `[-1]`).
- `outputs` declares a schema. The engine validates the task's produced object
  against it (via a generated Pydantic model), giving a precise error at the
  producing task instead of a cryptic failure at the consuming task.
- `over:` makes the `repeat_prompt` iterable explicit and typed, removing the
  "last tool result, double-json-decoded" heuristic. Legacy `repeat_prompt`
  without `over:` keeps the current behavior for one deprecation window.

### 5.3 Where pydantic-ai fits

The user specifically flagged pydantic-ai. Two viable strategies; the plan
recommends starting with (1) and treating (2) as an optional backend feature.

1. Pydantic-native structured outputs (recommended first step, no new hard
   dependency). Generate a Pydantic model from the `outputs` schema and use it
   to validate/coerce the producing task's result. This gives us typed,
   validated, named passing using pydantic (already a core dependency,
   `pydantic==2.13.3`) without adopting a second agent framework. It is fully
   under our control and testable offline.

2. pydantic-ai-powered structured extraction (optional, backend-scoped). For
   tasks that declare `outputs`, optionally drive the model with pydantic-ai's
   typed-output/result-validator machinery to have the model return a
   structured object directly (native tool/JSON-schema-constrained output)
   rather than us parsing free text. This is a natural new `AgentBackend`
   capability or an opt-in per-task `structured_output: true`. Because the
   backend layer is already abstracted (`sdk/base.py`), this can land as an
   adapter feature without disturbing existing backends. It does add
   `pydantic-ai` as a dependency, so it should be gated and optional.

Recommendation: land (1) to fix the grammar-level contract and de-risk cross
-task passing offline; evaluate (2) as a follow-up once the typed contract
exists, since (2) is an implementation detail of how a typed output gets
populated, not a grammar change.

### 5.4 Backwards compatibility

- `{{ result }}` and legacy `repeat_prompt` (implicit last-tool-result) keep
  working unchanged when `outputs`/`over`/`id` are absent.
- `outputs`/`id`/`over` are additive optional fields.
- The new `outputs.<id>` namespace is added alongside the existing `result`,
  `globals`, and `inputs` template namespaces in `template_utils.py`.
- Internally, the typed results store supersedes `last_mcp_tool_results` but we
  keep populating the old list during the deprecation window so resume/session
  behavior is unchanged.

### 5.5 Testing

- Schema generation: `outputs` spec -> Pydantic model -> validate good/bad
  payloads.
- Named passing: producing task registers `outputs.<id>`; consuming task
  renders from it; assert the exact rendered prompt.
- Fan-in: a fanned-out task's per-branch outputs are collectable as a list
  under `outputs.<id>` (closes section 2.2 fan-in gap).
- Regression: legacy `repeat_prompt` examples in `examples/taskflows/` still
  produce identical prompt sequences.

---

## 6. Additional enterprise-readiness improvements

These are independently shippable and can be sequenced around the two headline
features.

### 6.1 Optional strict grammar mode + taskflow linter

- Add a strict parsing mode that flips `extra="allow"` to `extra="forbid"` for
  the duration of validation, surfacing unknown-field errors with the offending
  key and document path. Off by default (backwards compatible); opt in via a
  `--strict` CLI flag and/or `SECLAB_TASKFLOW_STRICT=1`.
- Add a `lint` subcommand that validates every referenced document
  (taskflow, personalities, toolboxes, model_config, prompts) and reports
  unknown fields, unresolved model names, missing personalities/toolboxes, and
  template variables that will be undefined at render time, all without making
  a single model call. This is the highest-leverage enterprise change after the
  two headline features: it turns "expensive runtime failure" into "instant CI
  failure".

### 6.2 Structured run manifest and per-model results

- Emit a machine-readable run manifest (JSON) per run: resolved model IDs,
  per-model settings, backend, api_type, endpoint (redacted), start/stop
  timestamps, per-task and per-model status, and pointers to captured output
  artifacts.
- Extend `session.py` `CompletedTask` to hold a list of per-model results
  (status, model, backend, timing, optional token usage) instead of a single
  bool. Keep the bool as a derived property for compatibility.

### 6.3 JSON Schema export for authoring

- Add a `schema` subcommand that dumps `model_json_schema()` for each document
  type. Publish these so editors can validate and autocomplete taskflows.
  Wire into CI so schema drift is caught.

### 6.4 Grammar version and deprecation policy

- Document a policy: additive changes stay on `"1.0"`; behavior-changing or
  field-removing changes require a version bump with a deprecation window where
  both versions parse and deprecated fields emit warnings. Introduce a
  `SUPPORTED_VERSIONS` set (superseding the single `SUPPORTED_VERSION`) so more
  than one version can be accepted during a window.

### 6.5 Output artifacts on disk

- Persist each task/model output stream to a run-scoped artifacts directory
  (alongside sessions) so long audits are auditable and diffable after the
  fact, not just scrollback in a terminal. Ties into the manifest (6.2).

### 6.6 Concurrency isolation

- Remove the module-global `async_output` in favor of the `OutputRouter`
  (section 4.3) and give each run its own typed results store (section 5),
  eliminating shared mutable state as we add the second (model) concurrency
  axis. (Done: `render_utils.OutputRouter` owns the buffers, resolved via a
  `ContextVar` and set per run; the per-run `ResultStore` replaced the shared
  `last_mcp_tool_results` list.)

---

## 7. Backwards-compatibility and versioning strategy

- Every existing `examples/taskflows/*.yaml` must parse and run unchanged
  throughout. This is a hard CI gate: a test that loads and validates every
  bundled example on every PR.
- New fields are optional with today's behavior as the default.
- Pluralized fields define the singular as sugar, never remove it.
- Anything genuinely breaking is gated behind a version bump and a deprecation
  window, and is called out explicitly in `doc/MIGRATION.md`.
- The legacy `last_mcp_tool_results` channel is maintained in parallel with the
  typed results store until a version bump retires it.

---

## 8. Phased roadmap (mapped to PRs)

Each milestone is a self-contained, reviewable PR with tests and docs.

- M0 Foundations (no grammar change): extract `OutputRouter` and a per-run
  results store; land the `lint` and `schema` subcommands; add the "all
  bundled examples validate" CI gate. Pure hardening, unlocks everything else.
  (Largely delivered alongside M2 and a dedicated tooling slice: a neutral
  `ToolResult` + per-run `ResultStore` + single `decode_tool_result` replace the
  shared `last_mcp_tool_results` list and the faked copilot/anthropic envelope;
  `_fan_out_deploys` owns fan-out/concurrency/completion; an offline linter
  (`--lint`/`--strict`, `linting.py`) validates a taskflow and every referenced
  document with no model calls; `--schema` exports JSON Schema per document
  type; and a corpus gate (`tests/test_examples_validate.py`) validates every
  bundled document and lints every example taskflow on each test run. The
  module-global `async_output` is now encapsulated in an `OutputRouter`
  resolved via a `ContextVar` and set per run, so buffered async/multi-model
  streams are isolated. M0 complete.)
- M1 Multi-model, phase 1: DONE (branch `anticomputer/grammar-maturity`).
  `models:` (bare names or per-entry `{model, model_settings}` maps) with
  per-model parallel labelled streams, `model_concurrency`, and a
  `completion: all|any` policy; single-model path unchanged; multi-entry
  `models` + `repeat_prompt`/typed-outputs rejected for now. Full suite green +
  CI-parity lint clean + a live two-model CAPI run verified.
  (Feature A, sections 4.1-4.5.)
- M2 Typed outputs, phase 1: DONE (branch `anticomputer/grammar-maturity`).
  Built on a new I/O foundation: neutral `ToolResult` normalised across all
  three SDKs (`results.py`), per-run `ResultStore` (ordered results + named
  outputs, snapshot/restore for resume), and a single `decode_tool_result`.
  Grammar adds `id` + `outputs` (inline schema compiled to a Pydantic model in
  `output_schema.py`) + `over` (explicit typed iterable). A data-first Jinja
  environment makes keys like `items`/`keys` resolve to data. Legacy
  `repeat_prompt` unchanged (verified live). Full suite green + lint clean +
  live typed-outputs and legacy-repeat_prompt CAPI runs verified.
  (Feature B, strategy 1, section 5.)
- M3 Multi-model x repeat_prompt cross product, with explicit bounds and the
  fan-in from M2 to aggregate per-model/per-item results. DONE (branch
  `anticomputer/grammar-maturity`): `models` + `repeat_prompt` runs the item x
  model matrix concurrently, bounded by `model_concurrency * async_limit`, with
  per-branch stream labels `<model> [item <n>]`. `id` on a multi-model task
  fans in each branch's final result into `outputs.<id>` as a list of
  `{model, item, result}` records (pure `_aggregate_fanin`); branches use
  private tool-result sinks so the shared store stays deterministic. Covered by
  unit tests plus a run_main integration test (patched deploy) asserting the
  full matrix, labels, and fan-in, plus a live cross-product run.
- M4 Run manifest + per-model session results + on-disk artifacts. DONE
  (branch `anticomputer/grammar-maturity`): the session checkpoint gained
  per-task structured fields (models run against, wall-clock duration, skipped
  flag) and a `finished_at`; `TaskflowSession.manifest()` returns a curated,
  token-free machine-readable audit view (per-task status/models/timing plus
  named `outputs`, which carry per-model fan-in records), written to a
  run-scoped `artifacts/<id>/manifest.json` on finish/failure and printable via
  `--manifest <session_id>`. Per-model detailed results are the M3 fan-in
  records surfaced in the manifest. Covered by session, CLI, and run_main
  integration tests plus a live run. (Sections 6.2, 6.5. Bulk per-task output
  text artifacts deferred as a separate future slice to avoid reworking the
  streamed-output path.)
- M5 Optional pydantic-ai structured-output backend capability
  (Feature B, strategy 2, section 5.3) and strict-mode default for new grammar
  versions. (Section 6.1, 6.4.)

Ordering rationale: M0 de-risks concurrency and gives us CI safety nets before
we add axes of parallelism. M1 and M2 are independent and can be parallelized
across contributors. M3+ build on both.

---

## 9. Open questions to resolve during iteration

1. Multi-model x repeat_prompt: is the cross product the desired default, or
   should multi-model be disallowed on `repeat_prompt` tasks in phase 1 and
   only enabled in M3?
2. `completion` policy default for multi-model tasks: strict-all (proposed) vs
   any-success. Strict-all is safer but changes the "one failing model fails
   the task" ergonomics; confirm.
3. Output rendering default: interleaved-live vs buffered-then-flushed, and the
   N threshold to switch between them.
4. Typed outputs source of truth: inline `outputs` schema in the task (proposed)
   vs referencing a shared schema document (a new `filetype: schema`)? The
   latter is more reusable but adds a document type.
5. pydantic-ai adoption depth: grammar-level typed contracts only (strategy 1)
   vs also a pydantic-ai-driven backend (strategy 2), and whether strategy 2 is
   a new backend or a capability flag on existing backends.
6. Do we want per-model model_config selection (a task pointing at multiple
   model_config documents) or is per-entry override within one model_config
   sufficient?

---

## 10. Validation strategy for the whole effort

- Offline-first tests: every feature must be testable without live model calls,
  using the fake-backend and fake-stream patterns already in `tests/`
  (`test_sdk_*_adapter.py`, `test_stream.py`, `test_runner.py`).
- The "all bundled examples validate/parse" gate runs on every PR.
- Lint the repo with the project's configured ruff rules and run the full
  pytest suite locally before each push (per project contribution norms).
- Each grammar addition ships with: a model-level test, a runner behavior test,
  a docs update in `doc/GRAMMAR.md`, and at least one `examples/taskflows/`
  sample exercising it.

---

## 11. Follow-on work: typed outputs on JSON Schema

The `outputs` contract now uses standard JSON Schema (Draft 2020-12), validated
with the vendored `jsonschema` library, strictly and without coercion. That
validation is post-hoc only. The items below build on it, roughly in priority
order.

1. Schema-driven generation (highest leverage). Today we only validate after the
   fact, so the prompt must describe the shape in prose, which can drift from the
   declared schema. Feed the JSON Schema to each backend's native structured-
   output surface so the model is constrained to emit conforming output:
   - openai_agents: `response_format={"type": "json_schema", ...}` (OpenAI
     Structured Outputs), or an output tool whose parameters are the schema.
   - anthropic_sdk: a forced tool whose `input_schema` is the JSON Schema, or the
     native structured-output mode where available.
   - copilot_sdk: depends on what the session protocol exposes; fall back to
     validate-only if it cannot constrain generation.
   This removes prompt/schema drift and is the reason JSON Schema (not pydantic)
   was chosen: these APIs all consume JSON Schema.

2. Retry-to-repair on validation failure. Strict validation is only production-
   safe if a violation can be recovered. On a failed validation, re-prompt the
   model with the validation error (the pydantic-ai `ModelRetry` pattern) up to a
   small bound before failing the task/branch. This is a semantic retry, distinct
   from the existing transient-error retry loop. Suggested knob: `outputs_retries:
   N` (default 1 or 2).

3. Reusable schemas across tasks. `$ref`/`$defs` work within one schema, but a
   `Finding` shape re-declared in every task is duplication. Consider a first-
   class shared-schema document type (like model_configs) or a top-level
   `schemas:` block that tasks reference, so an audit corpus defines contracts
   once.

4. `format` assertion policy. jsonschema does not enforce `format` (e.g.
   `date-time`, `uri`) unless a FormatChecker is enabled. Decide whether audit
   contracts should assert formats; if so, enable `format_checker` on the
   validator. Document the pinned draft (2020-12) and the format policy.

5. Multi-model fan-in observability. A schema-violating branch currently records
   `result: null`. Consider surfacing the reason (e.g. an `error` field on the
   fan-in record) for debuggability, and add consensus helpers (majority vote /
   merge over per-model typed results), since ensemble is the point of typed
   multi-model outputs.

6. Author-facing error ergonomics. jsonschema ValidationError messages reference
   JSON-pointer paths; wrap them into friendlier "field X expected Y, got Z"
   messages for YAML authors, matching the linter's tone.

7. Coercion escape hatch (deliberately deferred). We chose strict validation so
   malformed model output surfaces instead of being silently reshaped. If real
   audit flows show frequent trivial type slips, an opt-in coercing mode could be
   added, but prefer items 1 and 2 over reintroducing lenient coercion.

8. Adoption note (not a migration). `outputs:` is introduced by this branch, so
   nothing pre-existing needs porting. The old bespoke-DSL form only ever existed
   on this branch and never shipped; this repo's examples/ were converted to JSON
   Schema here as internal cleanup. seclab-taskflows has no existing `outputs:` to
   rewrite; when it forward-ports to the matured grammar it adopts typed outputs
   (as JSON Schema) as new work.
