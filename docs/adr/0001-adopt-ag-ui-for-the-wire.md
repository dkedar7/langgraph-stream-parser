# ADR 0001 — Adopt AG-UI for the wire; narrow the core to the host layer

**Status:** Accepted — 2026-06-14
**Decision owner:** Kedar Dabhadkar

## Context

The LangStage family (six packages) shares `langgraph-stream-parser` as its
core. That package actually does four distinct jobs:

1. **Event translation** — typed `StreamEvent` dataclasses + `event_to_dict` +
   `StreamParser`, turning `graph.stream()` output into a wire format each
   surface renders.
2. **Tool-output extraction** — todos / reflection / display parsed out of tool
   results.
3. **Host config** — `HostConfig`, the `LANGSTAGE_*` / `langstage.toml`
   resolver.
4. **Agent loading** — `load_agent_spec` (the `module:attr` spec) + the demo
   stub.

We evaluated standardizing the *wire* on a popular agent protocol instead of
maintaining our own event vocabulary + per-surface serializers. **AG-UI**
(Agent-User Interaction Protocol, CopilotKit-stewarded, MIT, ~14k★, actively
maintained) is the agent-native fit: its ~16 event types map almost 1:1 onto
our `StreamEvent`s — including a first-class **interrupt/resume** path, the one
thing the OpenAI chat-completions spec could not represent.

Crucially, the official `ag-ui-langgraph` adapter is **MIT, standalone (no
CopilotKit runtime dependency), and already handles** interrupts+resume,
reasoning, tool calls, text, and state. Building our own AG-UI serializer would
reinvent it.

## Decision

1. **Adopt `ag-ui-langgraph` for the AG-UI wire.** Do not hand-write an AG-UI
   serializer.
2. **The core's durable identity is the host layer** (jobs 3 + 4: `HostConfig`
   + `load_agent_spec` + demo). That has *zero* overlap with AG-UI and is the
   family's actual differentiator ("one agent, every surface, same config").
   It stays, and is the thing we cherish.
3. **The event-translation layer (jobs 1 + 2) becomes legacy/optional.** It
   keeps working — surfaces still consume it today — but new integrations
   target AG-UI, and surfaces migrate off the bespoke event dicts *incrementally*.
4. **Additive now, destructive later.** Laying the AG-UI base additively is the
   small-impact-surface move; deleting the event layer in one swing would break
   all six surfaces at once (the opposite of the goal). The event layer is
   retired surface-by-surface only after AG-UI earns it.

## Consequences

- New module **`langgraph_stream_parser.agui`** + an `[agui]` extra
  (`ag-ui-langgraph[fastapi]`): given an agent spec (host layer) it stands up an
  AG-UI ASGI app. One blessed capability, reused by every surface — serve any
  LangStage agent over AG-UI with `langstage-agui --agent <spec>`.
- The wire stops being a differentiator (fine — it was always plumbing).
  Differentiation lives in the surfaces, the config story, and the future board.
- Dependency-risk: we cede the wire format to a CopilotKit-led spec. Accepted —
  MIT + 14k★ + multi-framework + daily commits make it a far safer bet than,
  e.g., the now-archived Open Agent Platform.
- The package name `langgraph-stream-parser` increasingly under-describes the
  keeper (the host layer). A future rename to a `langstage-core`-style identity
  is anticipated but **out of scope here** — we just shipped 0.3.0; no second
  rename churn yet.

## Not doing (now)

- Rewriting web/CLI/Jupyter/VS Code frontends to consume AG-UI — staged.
- Deleting the event layer / extractors — staged.
- Renaming the core package — later.
