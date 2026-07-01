# ADR 0003 — Deprecate and retire the event layer

**Status:** Proposed — 2026-07-01
**Decision owner:** Kedar Dabhadkar
**Builds on:** [ADR 0001](0001-adopt-ag-ui-for-the-wire.md), [ADR 0002](0002-execute-event-layer-retirement.md)

## Context

ADR 0002 is done: all four render surfaces (cli, jupyter, vscode, web) ship an
**opt-in** in-process AG-UI path, and the two shared mappings live in the core
(`agui.iter_event_frames`, `agui.iter_chunk_frames`). The dedupe is complete.

ADR 0002 listed the endgame as "deprecate `events.py` → remove at a major →
rename." That line was too glib. Executing it surfaces two facts that change the
plan, and this ADR exists to get them right *before* anything irreversible ships.

### Fact 1 — the AG-UI paths are opt-in; the event layer is still the default

Every surface still defaults to `StreamParser`. You cannot bolt a loud
`DeprecationWarning` onto `StreamParser.__init__` while it is the default code
path — it would fire on every normal run. **Deprecation must follow a
default-flip, not precede it.** (ADR 0002's remaining-work ordering had this
backwards.)

### Fact 2 — hermes is an un-migrated *fifth* surface, entangled with extractors

`langstage-hermes` was never migrated. Its `chat` command renders through
`StreamParser` + `PrintAdapter`, and — crucially — its entire reflection/skill
value rides on **three custom extractors** (`langstage_hermes/extractors.py`)
that implement `extractors.base.ToolExtractor` and emit `ToolExtractedEvent`s
(e.g. `skill_created`), which the cli renders as typed callouts.

So ADR 0002's "keep `extractors/`, remove the rest" is **incoherent**: extractors
are *driven by* `StreamParser` and *emit* an `events.py` type. AG-UI has no
extractor concept — extractors turn tool *results* into typed domain events, a
capability the AG-UI vocabulary doesn't carry. You cannot remove `StreamParser` /
`events.py` while keeping extractors functional for hermes.

## What "the event layer" actually is (inventory)

**Retire** (event-translation): `StreamParser`, the `events.py` dataclasses +
`event_to_dict`, `handlers/`, `stream_graph_updates` / `astream_graph_updates`,
and the render adapters `CLIAdapter` / `FastAPIAdapter` / `JupyterDisplay` /
`PrintAdapter`.

**Keep** (not event-translation, already load-bearing for AG-UI): the host layer
(`HostConfig`, `load_agent_spec`, demo), `tasks/`, the `agui/` module (`iter_*`,
`build_*`), `SessionAdapter` (now dual-mode — it keeps its AG-UI path), and the
input helper `prepare_agent_input` (the AG-UI paths reuse it for context-combining).

**Entangled** (the blocker): `extractors/` + hermes' dependency on
`StreamParser` + `PrintAdapter` + `ToolExtractedEvent`.

## Decision (proposed)

Retire the event layer in **five ordered stages**, gated so nothing user-visible
regresses and no warning fires on a default path.

1. **Solve the extractor → AG-UI story.** Add a way to run tool-result extractors
   over the AG-UI stream and surface their output — most likely mapping a matched
   extractor to an AG-UI `CustomEvent` (named, like `on_interrupt`) that
   `iter_event_frames` / `iter_chunk_frames` translate into a frame hermes'
   renderer understands. This is the one genuinely novel design piece; everything
   else is mechanical.
2. **Migrate hermes** (the fifth surface) onto the AG-UI path using (1), reaching
   parity for its reflection/skill callouts. Only after this do all five surfaces
   have an AG-UI path.
3. **Soak, then flip defaults.** Let the opt-in paths run in the wild; then flip
   each surface's default from `StreamParser` to AG-UI, one surface per release,
   each gated on parity, with the old path still reachable via an escape hatch
   (`--no-agui` / env) for one release.
4. **Deprecate.** Once nothing defaults to the event layer: `PendingDeprecationWarning`
   (silent) on the retire-set for one minor, then `DeprecationWarning` the next.
   Keep the extractor *protocol* (`ToolExtractor`) if (1) preserves it as the
   extension point; deprecate only the `StreamParser`-driven plumbing.
5. **Remove at the next major + rename.** Delete the retire-set and rename the
   core to a `langstage-core`-style identity in the same major, so users absorb
   one break, not two. `event_to_dict` frames remain producible (that shape is the
   AG-UI mapping's output) even though `StreamParser` is gone.

## Consequences

- **Longer than ADR 0002 implied.** This is a multi-release arc (hermes migration →
  default-flips × 5 → two-step deprecation → major). That is the cost of retiring a
  layer that five surfaces and one external-ish contract depend on, without churn.
- **hermes gets first-class AG-UI support** — the reflection/skill callouts, today
  a bespoke extractor path, become a documented AG-UI extension point. Net upgrade.
- The `event_to_dict` **wire shape survives** the death of `StreamParser` (it's what
  `iter_event_frames` emits), so external consumers of that JSON shape are unaffected
  by the internal retirement — only direct importers of `StreamParser`/`events` break,
  and only at the major, after two warning stages.

## Alternatives considered

- **Deprecate now, keep a frozen extractor sub-layer forever.** Rejected: it
  permanently keeps `StreamParser` + `events.py` + `extractors/` alive (most of the
  layer) to serve one surface — the opposite of the goal, and it strands hermes on
  the legacy path.
- **Drop hermes' extractor callouts.** Rejected: they are hermes' core value
  (reflection/skill visibility). Retiring infrastructure must not delete a feature.
- **Flip defaults before soaking.** Rejected: the opt-in paths are days old; flipping
  the default across five surfaces before real-world soak is how you ship a
  regression to everyone at once.

## Open questions / gates (must clear before Stage 3)

1. **Extractor → AG-UI fidelity.** Can a `CustomEvent`-based extractor bridge
   reproduce hermes' current callouts (`skill_created`, reflection, display) at
   parity? Prototype against hermes' three extractors before committing Stage 2.
2. **Escape hatch lifetime.** How long do we keep `--no-agui` after a default-flip?
   Proposed: one minor per surface, then remove with the event layer at the major.
3. **usage events.** ADR 0002 left token-usage parity as a spot-check; confirm the
   AG-UI paths carry it before flipping the surfaces that display cost.

## Not doing (now)

- Any `DeprecationWarning` on `StreamParser` / `events` — premature until Stage 4.
- The rename — bundled with the Stage 5 major.
- Removing `prepare_agent_input` or `SessionAdapter` — both are keepers.
