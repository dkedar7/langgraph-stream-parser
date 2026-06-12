# Deep Agent Runtime — Coordinated Revamp Plan

**Status:** all five packages built & tested — parser 0.2.0 (494), cowork-dash 0.4.0 (90), deepagent-lab 0.2.0 (51), deepagent-code 0.2.0 (18), deepagent-vscode 0.1.0 (9 + subprocess e2e; TS compiles). Four e2e examples pass; two real bugs found + fixed. **Wave 3 (coordinated publish) NOT yet executed — that's the remaining step and needs explicit go-ahead (irreversible: PyPI + Marketplace).**
**Author:** kzest (with Kedar)
**Date:** 2026-05-30
**Scope:** `langgraph-stream-parser`, `cowork-dash`, `deepagent-lab`, `deepagent-code`, `deepagent-vscode` (new)

---

## 1. Objective

Consolidate the duplicated agent-hosting plumbing that has accreted across the
deep-agent host projects into a single substrate — `langgraph-stream-parser` —
and ship every affected library in **one coordinated release window**.

The thesis from the audit: each host (`deepagent-lab`, `cowork-dash`,
`deepagent-code`) is the *same thing* — a surface that runs a user-supplied
LangGraph/deepagents `CompiledGraph` — wearing a different UI. They already
share an event vocabulary via `langgraph-stream-parser`, but each reimplements
agent loading, env-var config, a default agent, and (in `deepagent-lab`'s case)
an entire stale copy of the stream parser itself. This revamp removes that
duplication and adds a fourth surface (VS Code) on top of the cleaned-up core.

---

## 2. Release model — locked decisions

| Decision | Choice |
|---|---|
| **Version alignment** | **Parser leads, hosts pin.** The parser cuts one clean release; each host bumps independently but publishes in the same window. No forced shared version number. |
| **VS Code extension** | **In scope for this wave.** `deepagent-vscode` ships alongside the Python packages. |
| **Publish coordination** | Single release window, **dependency-ordered** publish. Parser to PyPI first; hosts follow once it is installable. |
| **Registries** | Two: **PyPI** (4 Python packages) + **VS Code Marketplace** (the extension). |

### Why "parser leads, hosts pin" rather than a shared 1.0

The hosts are still being actively reshaped; a synchronized `1.0` would be a
public API-stability promise we are not ready to make. Pinning
`langgraph-stream-parser>=0.2,<0.3` in each host gives us a coordinated wave
*and* room to break the substrate's API again in the next wave without a major
bump on every host.

### Target versions

| Package | Current | Target | Pins |
|---|---|---|---|
| `langgraph-stream-parser` | 0.1.9 → **0.2.0 (DONE)** | **0.2.0** | — |
| `cowork-dash` | 0.3.7 | **0.4.0** | `langgraph-stream-parser>=0.2,<0.3` |
| `deepagent-lab` | 0.1.4 (package.json) | **0.2.0** | `langgraph-stream-parser>=0.2,<0.3` |
| `deepagent-code` | 0.1.6 | **0.2.0** | `langgraph-stream-parser>=0.2,<0.3` |
| `deepagent-vscode` | — (new) | **0.1.0** | sidecar deps on `langgraph-stream-parser>=0.2,<0.3` |

All four Python packages carry the same git milestone tag:
**`unified-runtime-2026.05`**.

---

## 3. Audit summary (what exists today)

### `langgraph-stream-parser` is already a runtime kit, underadvertised

- **Events** (`events.py`): 14 typed events, each with `to_dict()`. This is the
  canonical wire schema. `InterruptEvent` carries the HITL decision helpers
  (`build_decisions`, `create_resume`, `allowed_decisions`).
- **Parser** (`parser.py`): sync + async, modes `updates`/`messages`/multi/`v2`/`auto`,
  subgraph namespace handling, pluggable extractor registry.
- **Adapters** (`adapters/`): `BaseAdapter` (state tracking + `run()` HITL loop),
  `PrintAdapter`, `CLIAdapter` (ANSI + spinner + arrow-key interrupts),
  `FastAPIAdapter` (stateless, per-session lock map, WebSocket + SSE),
  `JupyterDisplay`.
- **Extractors**: `ThinkToolExtractor`, `TodoExtractor`, `DisplayInlineExtractor`.
- **Resume** (`resume.py`): `create_resume_input`, `prepare_agent_input`
  (the latter already supports `context_parts`, currently unused by hosts).

### Duplication in the hosts

**`cowork-dash`** (faithful consumer, small deltas):

| Unit | Lines | Disposition |
|---|---:|---|
| `agent_loader.load_agent_from_spec` | 50 | → `host.load_agent_spec` |
| `config.AppConfig` / `from_env` | ~125 | shared keys → `host.HostConfig`; UI keys stay local |
| `default_agent.py` | ~160 | → `demo.create_default_agent` (extra) |
| `stream/sse_adapter.py` | 105 | **delete** → `SessionAdapter` |
| `stream/event_serializer.py` | 23 | **delete** → `event_to_dict(max_result_len=...)` |
| `stream/session_manager.py` | 76 | mostly **delete** → `SessionAdapter` |
| `server/routes_chat.py` | 164 | thin to ~50 lines wired to `SessionAdapter` |

**`deepagent-lab`** (heaviest divergence — runs a stale parallel parser):

| Unit | Lines | Disposition |
|---|---:|---|
| `langgraph_utils.py` | **543** | **delete almost entirely** — every function is reimplemented in the parser (the `write_todos` regex is byte-identical to `TodoExtractor`). Lab still uses the dict-based legacy API. |
| `agent_wrapper.py` | 409 | spec loader → `host.load_agent_spec`; keep context-injection + reload; route streaming through `StreamParser` |
| `config.py` | 107 | shared keys → `host.HostConfig`; lab keys (model, jupyter token, virtual mode) stay local |
| `agent.py` | 543 | default agent → `demo`; **keep** lab-specific notebook tools |
| `handlers.py` | 352 | rewrite Tornado handlers to consume parser events; reuse `to_dict()` wire shape |

**`deepagent-code`**: audited (Wave 0). **Not** a thin wrapper — it is a heavy
near-fork: `utils.py` (654 lines) reimplements the parser and `cli.py` (1555
lines) reimplements rendering + interrupts + spec loading. It does not depend on
`langgraph-stream-parser` yet. Its genuinely host-specific value is the
slash-command REPL shell (`CLIAdapter` has no multi-turn loop). Full delta in §6.4.

### Two corrections to the earlier proposal

1. **No SlashRegistry / WorkflowRegistry extraction.** These don't exist as
   backend code. "Slash commands" are a frontend autocomplete in cowork-dash
   that sends a configurable prompt template; "workflows" are markdown files the
   agent reads via filesystem tools. Both stay where they are.
2. **`SessionAdapter` is the real architectural addition**, not a lightweight
   helper. `FastAPIAdapter` is request-scoped; cowork-dash needs a session that
   survives disconnects, supports cancellation, and multiplexes parser events
   with side-channel events (file-change notifications). That's a new adapter.

---

## 4. Target architecture

```
                    ┌─────────────────────────────────────────────┐
                    │          langgraph-stream-parser 0.2.0       │
                    │                                              │
                    │  events/        ← canonical wire schema      │
                    │  parser/        ← StreamParser               │
                    │  extractors/    ← think/todo/display + custom │
                    │  adapters/                                    │
                    │    ├─ PrintAdapter   ├─ CLIAdapter            │
                    │    ├─ JupyterDisplay  ├─ FastAPIAdapter       │
                    │    └─ SessionAdapter  (NEW)                   │
                    │  host/          (NEW)                         │
                    │    ├─ load_agent_spec                         │
                    │    ├─ HostConfig                              │
                    │    └─ Workspace                               │
                    │  demo/          (NEW, [demo] extra)           │
                    │    └─ create_default_agent                    │
                    └───────┬───────────┬───────────┬──────────┬───┘
                            │           │           │          │
              ┌─────────────┘   ┌───────┘    ┌──────┘    ┌─────┘
              ▼                 ▼            ▼            ▼
       ┌─────────────┐  ┌──────────────┐ ┌──────────┐ ┌──────────────┐
       │ cowork-dash │  │ deepagent-lab│ │ deepagent│ │ deepagent-   │
       │   0.4.0     │  │  next minor  │ │  -code   │ │  vscode 0.1.0│
       │ (web/FastAPI│  │ (JupyterLab  │ │ (CLI)    │ │ (TS ext +    │
       │  + React)   │  │  extension)  │ │          │ │  py sidecar) │
       └─────────────┘  └──────────────┘ └──────────┘ └──────────────┘
```

**The wire contract.** `event.to_dict()` is the single inter-process protocol.
Every surface — FastAPI WebSocket/SSE, Jupyter handler, CLI, and the VS Code
stdio sidecar — serializes the same event dicts. The TS extension's event
dispatcher pattern-matches on the same `type` field the web frontend already
uses.

---

## 5. Shared API surface — `langgraph-stream-parser 0.2.0` additions

### 5.1 `host.load_agent_spec`

```python
def load_agent_spec(spec: str) -> Any:
    """Load a LangGraph agent from a 'path.py:var' or 'module:var' spec.

    Strict: the ':var' suffix is required. No implicit 'agent'/'graph'
    fallback (that is where lab and dash diverged).

    Raises ValueError on malformed spec, FileNotFoundError on missing file,
    AttributeError on missing variable.
    """
```

Replaces `cowork_dash/agent_loader.py` and the loader half of
`deepagent_lab/agent_wrapper.py`. **Behavior decision: strict mode wins** —
explicit over implicit. `deepagent-lab` loses its `agent`→`graph` fallback.

### 5.2 `host.HostConfig`

```python
@dataclass
class HostConfig:
    """Shared DEEPAGENT_* env-var schema. Hosts subclass to add their own keys."""
    agent_spec: str | None = None        # DEEPAGENT_AGENT_SPEC
    workspace_root: Path = Path(".")      # DEEPAGENT_WORKSPACE_ROOT
    host: str = "localhost"               # DEEPAGENT_HOST
    port: int = 8050                      # DEEPAGENT_PORT
    debug: bool = False                   # DEEPAGENT_DEBUG
    title: str = "Deep Agent"             # DEEPAGENT_TITLE

    @classmethod
    def from_env(cls) -> "HostConfig": ...
    def merge(self, **overrides) -> "HostConfig": ...
```

Only the **intersection** of the hosts' schemas. UI-specific keys
(`THEME`, `AUTH_*`, `ICON_URL`, canvas/files toggles, model name, jupyter token,
virtual mode) **stay in each host's own config subclass**. Drift is acceptable
below the shared core.

### 5.3 `host.Workspace`

```python
@dataclass
class Workspace:
    root: Path
    def ensure(self) -> "Workspace": ...        # mkdir -p root
    def subpath(self, *parts) -> Path: ...        # safe join under root
```

Small convenience wrapper. Not load-bearing.

### 5.4 `event_to_dict(event, *, max_result_len=...)`

Add the `max_result_len` knob to `event_to_dict` (today it only exists on
`ToolCallEndEvent.to_dict`). This lets cowork-dash **delete `EventSerializer`
entirely** and call the library function directly.

### 5.5 `adapters.SessionAdapter` (the meaty new piece)

```python
class SessionAdapter:
    """Stateful, session-scoped streaming for web hosts.

    Beyond FastAPIAdapter:
    - Per-session event queue (server pushes; transport drains)
    - Cancellation of an in-flight turn
    - Side-channel push_event(dict) — e.g. file-change notifications
      multiplexed into the same stream as parser events
    - Reconnect: a session survives client disconnect; SSE resumes
    """

    def __init__(self, *, graph, stream_mode=["updates", "messages"], **parser_kwargs): ...
    def get_or_create(self, session_id: str | None = None) -> "Session": ...
    async def run_turn(self, session_id: str, message: str, *, context_parts=None): ...
    async def resume(self, session_id: str, decisions: list[dict]): ...
    def cancel(self, session_id: str) -> bool: ...
    async def push_event(self, session_id: str, event: dict) -> None: ...
    async def sse(self, session_id: str): ...   # async generator of "data: {...}\n\n"
```

Absorbs `cowork_dash/stream/sse_adapter.py`, `session_manager.py`, and the bulk
of `routes_chat.py`. Uses `prepare_agent_input(context_parts=...)` so the
hosts stop hand-rolling "[Current time: ...]" injection.

### 5.6 `demo.create_default_agent` (`[demo]` extra)

```python
# pip install langgraph-stream-parser[demo]   →   requires deepagents
def create_default_agent(workspace, *, model="anthropic:claude-sonnet-4-6", tools=None): ...
```

One canonical default agent (filesystem backend + think/todo + sensible system
prompt). Lazy-imports `deepagents` so the base package stays dependency-light.
Replaces `cowork_dash/default_agent.py` and `deepagent_lab/agent.py`'s default —
**but each host keeps its own domain tools** (cowork's notebook/canvas tools,
lab's nbformat tools) by passing `tools=`.

---

## 6. Per-package workstreams

### 6.1 `langgraph-stream-parser` → 0.2.0 — **IMPLEMENTED**

- [x] Add `host/` submodule: `load_agent_spec`, `HostConfig`, `Workspace`.
- [x] Add `max_result_len` to `event_to_dict`.
- [x] Add `adapters/session.py`: `SessionAdapter` + `Session`.
- [x] Add `demo/` submodule + `[demo]` optional extra in `pyproject.toml`.
- [x] Wire `prepare_agent_input(context_parts=...)` into `SessionAdapter`.
- [x] Tests: spec loader (file/module/error paths), `HostConfig.from_env`,
      `SessionAdapter` (queue, cancel, reconnect, side-channel), `demo` smoke,
      wire-contract golden file. **54 new tests; 492 total passing.**
- [ ] README rewrite: reposition as "runtime kit", document `host`/`demo`/`SessionAdapter`.
- [ ] CHANGELOG: from this release on, strict semver.
- [ ] Publish `0.2.0rc1` to TestPyPI.

> Note: `SessionAdapter.push_event` is **synchronous** (unbounded queue +
> `put_nowait`), not async as the §5.5 sketch suggested — safer inside
> cancellation/except blocks and simpler for callers.

> **Bug found by end-to-end examples (fixed):** `skip_tools` in the updates
> handler returned *before* running a tool's extractor, so
> `compat.stream_graph_updates(skip_tools=["write_todos","think_tool"])`
> silently dropped `todo_list` and `reflection` updates — which would have
> broken deepagent-lab's todo sidebar after migration. Unit tests missed it
> (they mock `stream_graph_updates`); the e2e example caught it. Fix:
> `handlers/updates.py` now runs the extractor first and only suppresses the
> *lifecycle* end event for skipped tools (extractors exist precisely to
> surface skipped-tool content). Added two regression tests to `test_compat.py`
> driving `write_todos`/`think_tool` through the full path. Parser now 494 tests.

> **End-to-end validation:** `examples/cowork_e2e.py` drives the real FastAPI
> app → `/api/chat` route → `SessionAdapter` → `adapter.sse()` and asserts the
> full `session_init → content → tool_start → tool_end → complete` frame
> sequence; `examples/lab_e2e.py` drives `AgentWrapper.execute` through compat
> and asserts the `chunk / tool_calls / todo_list / complete` dict shapes the TS
> frontend reads. Both pass with deterministic fake agents (no model/network).

### 6.2 `cowork-dash` → 0.4.0 — **IMPLEMENTED**

- [x] **Delete:** `agent_loader.py`, `stream/event_serializer.py`,
      `stream/sse_adapter.py`, `stream/session_manager.py` (whole files removed;
      `stream/` now holds only an empty `__init__.py`).
- [x] `app.py` → uses `langgraph_stream_parser.load_agent_spec`.
- [x] `server/main.py` → one `SessionAdapter(graph=agent, stream_mode=["updates","messages"], max_result_len=50_000, **stream_parser_config)`; `app.state.session_adapter` replaces `session_manager`.
- [x] `server/routes_chat.py` → thin wiring to `SessionAdapter`; file-watcher kept as a `push_event` producer; `context_parts()` helper carries the time+cwd injection that used to live in `sse_adapter`.
- [x] `server/routes_session.py` → `list/delete/inject` on the adapter.
- [x] `default_agent.py` → builds on `demo.create_default_agent` (passes prompt, notebook/display tools, `CanvasMiddleware`, `interrupt_on={"bash": True}`); keeps `AGENT_TOOLS`/`AGENT_MIDDLEWARE`/`.middleware` pin for canvas auto-detection.
- [x] Bump to 0.4.0; pin `langgraph-stream-parser>=0.2,<0.3`.
- [x] Tests: deleted the 4 tests for deleted modules; rewrote `test_inject.py` on the adapter; added `test_streaming.py` preserving the library-drift regression guards. **90 tests passing against local parser 0.2.0.**
- [ ] Verify SSE protocol unchanged from the *frontend's* perspective in a real browser (event dicts are identical by construction — `event_to_dict` is the same function the old `EventSerializer` delegated to — but not yet exercised end-to-end in the UI).

> **Deviation from the original plan:** `config.py` was **not** rebased onto
> `HostConfig`. `AppConfig` stays standalone. Rationale: the cowork tests lock
> its `workspace` field name (vs `HostConfig.workspace_root`) and its dict-based
> `merge`, so subclassing would force renames + test churn to deduplicate only
> ~5 trivial scalar fields — negative value. The high-value dedup (agent
> loading, streaming/session machinery, event serialization, default-agent
> boilerplate) all landed; the config dataclass was never the duplication worth
> chasing.

> **Pre-existing warning (not a regression):** `create_deep_agent` emits a
> `model=None` deprecation warning because cowork's default agent never
> specified a model (it relied on deepagents' default). The old code did the
> same; choosing an explicit default model is a separate cowork product decision.

### 6.3 `deepagent-lab` → 0.2.0 — **IMPLEMENTED (via compat, NOT a frontend remap)**

**Key decision — the frontend was NOT touched.** The plan feared a high-risk TS
remap from the dict shape to typed `to_dict()`. On inspection, the parser's
**`compat` layer reproduces lab's exact dict shape** — `compat.py`'s docstring
literally calls lab's `langgraph_utils` the "reference implementation" it was
built to match. Verified key-by-key against the frontend (`widget.tsx`):
`data.chunk` / `data.tool_calls` / `data.todo_list` / `data.status` /
`data.interrupt`, and the interrupt UI's `action_requests[0].tool` +
`review_configs[0].allowed_decisions` (the parser's `process_interrupt`
normalizes `name`→`tool`, confirmed at `handlers/updates.py:81`). So routing lab
through `compat.stream_graph_updates` is a drop-in — **zero frontend risk.** The
typed-event remap is deferred to a later, optional improvement.

- [x] **Delete:** `langgraph_utils.py` (543-line stale parser) + its test.
- [x] `agent_wrapper.py` → imports `stream_graph_updates`/`prepare_agent_input`
      from `langgraph_stream_parser` (compat); `_load_agent` rewritten onto
      `host.load_agent_spec` (the two `_load_agent_from_*` helpers deleted),
      keeping the `agent`→`graph` default-name fallback. Context-injection,
      reload, set_root_dir, execute all unchanged.
- [x] `agent.py` → default agent built via `demo.create_default_agent`
      (passes lab's nbformat notebook tools + system prompt + `chat_model`);
      verified it still compiles to a `CompiledStateGraph` named "Default Agent".
- [x] `handlers.py` → unchanged (Tornado thread-pool/queue bridge still valid;
      it consumes `agent_wrapper`, which now routes through the shared parser).
      The `{"status":"cancelled"}` event stays here.
- [x] **TS frontend: untouched.**
- [x] Add `langgraph-stream-parser>=0.2,<0.3` to `pyproject.toml` (lab did not
      depend on the parser at all before); bump `package.json` 0.1.4 → 0.2.0.
- [x] Tests: deleted `test_langgraph_utils.py`; the rest unchanged. **51 tests passing** (`test_agent_wrapper`, `test_config`, `test_launcher`) against local parser 0.2.0.

> **Deviation from the original plan (for the better):** §6.3 prescribed a typed-event
> frontend remap as "the highest-risk item in the wave." It turned out to be
> unnecessary — `compat` exists precisely to preserve this shape. Risk avoided,
> ~540 lines of stale parser still deleted, goal met.

### 6.4 `deepagent-code` → 0.2.0

**Audited (Wave 0 complete).** Cloned to `C:\Users\Kedar\Documents\Code\deepagent-code`
(HEAD `e58a9f2`). Tiny package, 4 source files, but a **heavy near-fork** of the
parser — it predates the extraction and does **not** depend on
`langgraph-stream-parser` at all.

| Unit | Lines | Disposition |
|---|---:|---|
| `deepagent_code/utils.py` | 654 | **Delete.** Reimplements the whole parser: `stream_graph_updates`/`astream_graph_updates`, interrupt parsing, todo/think extraction, tool-call serialization. Replace call sites with `StreamParser` + `compat`. |
| `deepagent_code/cli.py` | 1555 | **Largest change.** Replace `parse_agent_spec`/`load_graph*` (lines ~406–542) with `host.load_agent_spec`; replace the ANSI/`Spinner`/`render_markdown`/arrow-key interrupt stack and single-turn rendering with `CLIAdapter`. **Keep** the slash-command registry, `run_conversation_loop`, and `!bash` passthrough. |
| `deepagent_code/config.py` | 114 | Replace with a `HostConfig` subclass (add `stream_mode`, `graph_name`, `verbose`, `async_mode`). Its TOML layer (`~/.deepagents/config.toml` + project `deepagents.toml`, deep-merge) has no `host` equivalent — keep TOML resolution inside the subclass. |

Two specific gotchas surfaced by the audit:
- **`CLIAdapter` has no multi-turn REPL loop.** `BaseAdapter.run()` drives a
  *single* input to completion (with interrupt resumption), not an interactive
  shell. So `deepagent-code`'s `run_conversation_loop` (the multi-turn REPL with
  slash commands and `!bash`) is genuinely host-specific and **stays**. The
  adapter renders one turn's events + handles its interrupt prompt; the loop
  wraps it.
- **Env-var name drift.** `deepagent-code` reads `DEEPAGENT_SPEC` (plus legacy
  `DEEPAGENT_AGENT_SPEC`); `HostConfig` standardizes on `DEEPAGENT_AGENT_SPEC`.
  Migration reconciles to the standard name (keep `DEEPAGENT_SPEC` as a
  deprecated alias for one release).
- **Spec-loader tightening.** `deepagent-code` accepts a bare path with a default
  `graph` object name; `host.load_agent_spec` requires the explicit `:object`
  suffix. Migration tightens accepted input — document it in the changelog.

**IMPLEMENTED (via compat, like lab — no CLIAdapter rendering swap):**
- [x] Add `langgraph-stream-parser>=0.2,<0.3` to `pyproject.toml`; bump version
      0.1.6 → 0.2.0 (pyproject + `cli.py:58 __version__` + `__init__.py`).
- [x] **Delete `utils.py`** (654 lines) + `test_utils.py`. `cli.py` now imports
      `prepare_agent_input`/`stream_graph_updates`/`astream_graph_updates` from
      `langgraph_stream_parser` (compat). Verified `print_chunk` reads the same
      dict keys compat emits (`chunk`/`tool_calls`/`node`/`interrupt`/`error`);
      its `tool_result`/`todo_list` branches were already dead code, so no
      visible CLI change.
- [x] `cli.py` `load_graph` → thin wrapper delegating to `host.load_agent_spec`;
      deleted `load_graph_from_file`/`load_graph_from_module`. Kept
      `parse_agent_spec` (only `test_cli` uses it).
- [x] `__init__.py` → re-exports the parser's public streaming API (dropped the
      incidental re-export of parser internals — deepagent-code is a CLI).
- [x] Tests: deleted `test_utils.py`; `test_cli` + `test_config` unchanged.
      **18 passing** against local parser 0.2.0. End-to-end example
      (`examples/deepagent_code_e2e.py`) drives `load_graph` + `run_single_turn_sync`
      through compat and passes.

> **Deviations from the original plan:**
> 1. **`config.py` kept as-is** (not a `HostConfig` subclass). It's a generic
>    TOML+env *resolver* (deep-merge, dotted-key, CLI>env>TOML precedence), not a
>    config dataclass — it doesn't duplicate `HostConfig` and the TOML layer has
>    no host equivalent. Subclassing would lose it.
> 2. **CLI rendering kept** (no `CLIAdapter` swap). Like lab's frontend, the
>    rendering stack (`print_chunk`, spinner, slash-command REPL, `!bash`) is
>    bespoke and high-risk to replace; compat preserves its dict contract, so the
>    654-line `utils.py` duplication is gone without touching the UX. The
>    `CLIAdapter` adoption is deferred to a later optional pass.
>
> **Bug found + fixed by the example:** `load_graph`'s `:`-detection mistook a
> Windows drive-letter colon (`C:\...\agent.py`) for a `:graph_name` suffix,
> splitting the path at `C:`. Pre-existing in the original `load_graph` too.
> Fixed: only treat a trailing `:token` as a graph name when it has no path
> separator. The `DEEPAGENT_SPEC` → `DEEPAGENT_AGENT_SPEC` reconciliation is
> deferred (CLI-specific env var; not load-bearing for the release).

### 6.5 `deepagent-vscode` → 0.1.0 (new repo) — **BUILT** (sidecar tested; extension compiles, not yet run in VS Code)

New repo at `C:\Users\Kedar\Documents\Code\deepagent-vscode` (git-init'd, staged,
uncommitted). PyPI package name decision: **`deepagent-vscode`** (module
`deepagent_vscode`, console script `deepagent-vscode-sidecar`).

**(a) Python sidecar** (`deepagent_vscode/sidecar.py`) — **done + tested**:
- [x] `load_agent_spec` + `StreamParser`; reads NDJSON commands on stdin, emits
      `event.to_dict()` NDJSON on stdout. Depends directly on
      `langgraph-stream-parser>=0.2,<0.3` (not on `deepagent-code`).
- [x] Interrupt round-trip on the same channel (`{"type":"decision",...}` inbound).
- [x] Protocol: `ready`/`ack`/`<events>`/`complete`/`turn_end`; commands
      `message`/`decision`/`shutdown`. `run()` core is stream-injectable for tests.
- [x] **9 unit tests** (`tests/test_sidecar.py`) + a **subprocess e2e example**
      (`examples/vscode_sidecar_e2e.py`) that spawns `python -m deepagent_vscode`
      and pipes real stdio — both pass.

**(b) VS Code extension** (`extension/`, TypeScript) — **scaffolded + compiles**:
- [x] Registers a chat participant (`@deepagent`) via `vscode.chat.createChatParticipant`.
- [x] Spawns + supervises the sidecar; one `dispatch()` pattern-matching on
      `event.type` → chat stream (content / reasoning / tool_start / tool_end /
      todos / error). Same event vocabulary as every other surface.
- [x] Config: `deepagent.agentSpec` + `deepagent.pythonPath` settings.
- [x] `npm install` + `tsc` compile clean → `dist/extension.js`.
- [ ] **Interrupt confirmation UI** — v0 surfaces the pending action but does not
      yet send a `decision` back (the sidecar supports it; the chat-UI affordance
      is the next increment).
- [ ] **Not yet run in an Extension Development Host** — the participant
      registration + sidecar spawn + dispatch type-check, but haven't been
      exercised in a live VS Code session (the GUI-equivalent of the
      browser/JupyterLab caveats for cowork/lab).

**Release channel note:** the extension publishes to **VS Code Marketplace**
(via `vsce publish`), the sidecar to **PyPI**. The coordinated wave therefore
touches two registries — sequence the sidecar (PyPI) before the extension.

---

## 7. Release mechanics — the single window

Even a coordinated publish has an unavoidable order: the parser must be
installable before the hosts can pin it.

### Pre-release (can overlap across repos)
1. Feature branch per repo: `unified-runtime`.
2. Integration-test hosts against the parser via **editable install**:
   `uv pip install -e ../langgraph-stream-parser` in each host's central venv.
3. Publish parser **0.2.0rc1 to TestPyPI**; install each host against it
   (`uv pip install -i https://test.pypi.org/simple/ langgraph-stream-parser==0.2.0rc1`).
4. Green light: all host test suites pass against the RC; manual UI smoke for
   cowork-dash (web), deepagent-lab (JupyterLab), deepagent-vscode (extension host).

### Release day (dependency-ordered)
5. **PyPI publish order:**
   a. `langgraph-stream-parser==0.2.0` → wait for index availability.
   b. In parallel once (a) is live: `cowork-dash`, `deepagent-lab`,
      `deepagent-code`, `deepagent-vscode` sidecar — each pinning `>=0.2,<0.3`.
6. **VS Code Marketplace:** `vsce publish` the extension (after its sidecar is on PyPI).
7. **Tag** all four Python repos + the vscode repo with `unified-runtime-2026.05`.
8. **Smoke after publish:** fresh venv, `pip install` each host from PyPI, run the
   golden path (one prompt → one tool call → one HITL approval) on every surface.

### Rollback
- Each package is independently yankable on PyPI. Because hosts pin a range, a
  bad parser `0.2.0` can be yanked and republished as `0.2.1` without touching
  host pins. Keep RC verification strict to avoid needing this.

---

## 8. Testing strategy

| Layer | What |
|---|---|
| Parser unit | spec loader paths, `HostConfig.from_env`, `SessionAdapter` queue/cancel/reconnect/side-channel, `demo` smoke |
| Wire-contract | golden-file test: every `event.to_dict()` shape snapshotted; hosts assert against the same snapshots so frontend mappings can't silently drift |
| Host integration | each host's suite run against the parser RC from TestPyPI |
| UI smoke (manual) | cowork-dash in browser; deepagent-lab in JupyterLab; deepagent-vscode in an Extension Development Host |
| Post-publish | clean-venv `pip install` + golden path per surface |

The **wire-contract golden-file test is the linchpin** — it's what lets four
independently-versioned surfaces trust the same event shapes.

---

## 9. Risks & open items

1. **deepagent-lab frontend migration (highest risk).** The TS sidebar consumes
   the legacy dict shape. Re-mapping it to the typed `to_dict()` shape is the
   single largest and least-reversible task. Mitigation: do it first within the
   lab workstream, behind the RC, with the golden-file snapshots as the spec.
2. **`SessionAdapter` correctness.** Concurrency (per-session locks), cancel
   semantics, and reconnect replay are easy to get subtly wrong. Heavy unit
   coverage required before cowork-dash leans on it.
3. **Two-registry release.** PyPI + Marketplace in one window adds moving parts.
   Sequence strictly; the sidecar gates the extension.
4. **deepagent-code is unaudited.** Wave 0 must clone and confirm it really is a
   thin CLIAdapter wrapper; if it diverged like lab did, its workstream grows.
5. **Parser scope creep.** With `host` + `demo` + `SessionAdapter`, the package
   is now a small framework. README must reframe it, and semver discipline
   starts at 0.2.0.
6. **Default-agent coupling.** `demo` pins a `deepagents` range via the `[demo]`
   extra. Bump deliberately, not reactively.

### Decisions still needed
- [x] ~~deepagent-lab and deepagent-code current versions~~ → 0.1.4 and 0.1.6 (audited).
- [x] ~~Whether `demo` ships notebook tools~~ → **filesystem-only**, hosts inject
      their own tools via `tools=` (implemented in 0.2.0).
- [ ] Sidecar package name: `deepagent-vscode` on PyPI vs a `-sidecar` suffix.
- [ ] `DEEPAGENT_SPEC` → `DEEPAGENT_AGENT_SPEC` reconciliation: confirm we keep
      the old name as a deprecated alias for one release (recommended).

---

## 10. Sequenced checklist

**Wave 0 — prep**
- [x] Clone + audit `deepagent-code`; finalize §6.4.
- [x] Record current versions; confirm target version numbers.
- [ ] Cut `unified-runtime` branches in all repos; create `deepagent-vscode` repo.

**Wave 1 — parser**
- [x] Implement §5 (host, demo, SessionAdapter, event_to_dict knob).
- [x] Tests (54 new, 492 total passing) + wire-contract golden file.
- [ ] README rewrite.
- [ ] Publish `0.2.0rc1` to TestPyPI.

**Wave 2 — hosts (parallel, against the RC)**
- [x] cowork-dash migration (§6.2) — **done, 90 tests passing vs local parser 0.2.0**.
- [x] deepagent-lab migration (§6.3) — **done via compat, no frontend remap; 51 tests passing**.
- [x] deepagent-code migration (§6.4) — **done via compat, no CLIAdapter swap; 18 tests passing + e2e example**.
- [x] deepagent-vscode sidecar + extension (§6.5) — **sidecar done (9 tests + subprocess e2e); extension compiles, not yet run in VS Code**.

**Wave 3 — coordinated publish**
- [ ] Parser 0.2.0 → PyPI.
- [ ] Hosts → PyPI (pinned).
- [ ] Extension → Marketplace.
- [ ] Tag `unified-runtime-2026.05` across repos.
- [ ] Post-publish golden-path smoke on all surfaces.
```
