"""Serve any LangGraph agent over the **AG-UI** protocol.

AG-UI (Agent-User Interaction Protocol) is the event-based wire format for
streaming rich agent interactions — text, tool calls, reasoning, state, and
human-in-the-loop interrupts — to frontends. This module is the LangStage
family's blessed bridge to it: the host layer (``load_agent_spec`` +
``HostConfig``) resolves *which* agent to run, and the official, MIT-licensed
``ag-ui-langgraph`` adapter owns the wire. The result is that every surface's
agent is reachable from any AG-UI client without each surface reimplementing a
protocol.

See ``docs/adr/0001-adopt-ag-ui-for-the-wire.md`` for the rationale.

Requires the ``agui`` extra::

    pip install "langgraph-stream-parser[agui]"

Quick start::

    # Serve any agent spec over AG-UI:
    langstage-agui --agent my_agent.py:graph

    # Or in code:
    from langgraph_stream_parser.agui import build_app
    app = build_app(my_compiled_graph)   # an ASGI app; run with uvicorn
"""
# NB: intentionally NOT `from __future__ import annotations`. The resilient
# endpoint below needs real (non-string) annotations so FastAPI can resolve
# RunAgentInput as the request body; PEP 604 unions work natively on >=3.11.
from typing import Any

__all__ = [
    "build_agent",
    "add_agui_endpoint",
    "build_app",
    "serve",
    "ensure_available",
    "DEFAULT_AGENT_NAME",
]

DEFAULT_AGENT_NAME = "LangStage Agent"

_IMPORT_HINT = (
    "AG-UI support needs the 'agui' extra: "
    'pip install "langgraph-stream-parser[agui]"'
)


def ensure_available() -> None:
    """Raise the agui-extra ``RuntimeError`` if the AG-UI server deps are missing.

    Lets a caller fail fast with the clean install hint *before* any user-facing
    output (e.g. a "Serving … at <url>" banner), instead of mid-serve.
    """
    try:
        import ag_ui_langgraph  # noqa: F401
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:  # pragma: no cover - only without the [agui] extra
        raise RuntimeError(_IMPORT_HINT) from e


def _is_langgraph_agent(obj: Any) -> bool:
    """True if obj is already an ``ag_ui_langgraph.LangGraphAgent`` (has the
    adapter contract), so callers may pass a prebuilt agent through."""
    return hasattr(obj, "clone") and hasattr(obj, "run") and hasattr(obj, "name")


def build_agent(
    graph: Any,
    *,
    name: str = DEFAULT_AGENT_NAME,
    description: str | None = None,
    config: Any = None,
) -> Any:
    """Wrap a compiled LangGraph graph in an ``ag-ui-langgraph`` ``LangGraphAgent``.

    Args:
        graph: A compiled LangGraph graph (``CompiledStateGraph``). Its state
            must include a ``messages`` key (the AG-UI adapter's only schema
            requirement) — true for ``MessagesState`` and deepagents graphs.
        name: Display name surfaced to AG-UI clients.
        description: Optional human description.
        config: Optional ``RunnableConfig`` / dict forwarded to the graph.

    Returns:
        A ``LangGraphAgent`` ready to attach to an ASGI app.

    Raises:
        RuntimeError: if the ``agui`` extra is not installed.
    """
    try:
        from ag_ui_langgraph import LangGraphAgent
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_IMPORT_HINT) from e
    # AG-UI requires threaded state — the adapter calls graph.aget_state() and
    # supports interrupts/resume, both of which need a checkpointer. Many user
    # graphs are compiled without one (and would otherwise hard-crash with
    # "No checkpointer set"), so attach an in-memory default when absent.
    if getattr(graph, "checkpointer", None) is None:
        try:
            from langgraph.checkpoint.memory import InMemorySaver

            graph.checkpointer = InMemorySaver()
        except Exception:  # pragma: no cover - best-effort; LangGraphAgent will surface real issues
            pass
    return LangGraphAgent(name=name, graph=graph, description=description, config=config)


def add_agui_endpoint(
    app: Any,
    graph: Any,
    *,
    path: str = "/",
    name: str = DEFAULT_AGENT_NAME,
    description: str | None = None,
    config: Any = None,
) -> Any:
    """Attach an AG-UI endpoint for ``graph`` to an existing FastAPI ``app``.

    ``graph`` may be a compiled graph or an already-built ``LangGraphAgent``.
    Returns the same ``app`` for chaining.

    The endpoint is *resilient*: if the agent raises mid-run, a terminal
    ``RUN_ERROR`` event is emitted and the stream closes cleanly, rather than
    crashing the connection with an unhandled 500 (the bare upstream adapter
    lets node exceptions propagate). Each request runs on its own cloned agent.
    """
    try:
        from ag_ui.core import EventType, RunAgentInput, RunErrorEvent
        from ag_ui.encoder import EventEncoder
        from fastapi import Request
        from fastapi.responses import StreamingResponse
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(_IMPORT_HINT) from e

    agent = graph if _is_langgraph_agent(graph) else build_agent(
        graph, name=name, description=description, config=config
    )

    @app.post(path)
    async def _run(input_data: RunAgentInput, request: Request):
        accept = request.headers.get("accept")
        try:
            encoder = EventEncoder(accept=accept)
        except TypeError:  # pragma: no cover - older SDKs without the accept kwarg
            encoder = EventEncoder()
        media_type = getattr(encoder, "get_content_type", lambda: "text/event-stream")()
        run_agent = agent.clone()

        async def gen():
            try:
                async for ev in run_agent.run(input_data):
                    # run() yields SSE-encoded strings; encode objects defensively.
                    yield ev if isinstance(ev, (str, bytes)) else encoder.encode(ev)
            except Exception as exc:  # noqa: BLE001 - surfaced to the client as RUN_ERROR
                yield encoder.encode(
                    RunErrorEvent(
                        type=EventType.RUN_ERROR,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )

        return StreamingResponse(gen(), media_type=media_type)

    @app.get(path)
    async def _health():
        return {"status": "ok", "agent": {"name": getattr(agent, "name", name)}}

    return app


def build_app(
    graph: Any,
    *,
    path: str = "/",
    name: str = DEFAULT_AGENT_NAME,
    description: str | None = None,
    config: Any = None,
    title: str | None = None,
) -> Any:
    """Build a standalone FastAPI ASGI app exposing ``graph`` over AG-UI.

    Run it with any ASGI server, e.g. ``uvicorn.run(app, ...)`` — or just use
    :func:`serve`.
    """
    try:
        from fastapi import FastAPI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(_IMPORT_HINT) from e
    app = FastAPI(title=title or name)
    add_agui_endpoint(app, graph, path=path, name=name, description=description, config=config)
    return app


def serve(
    spec_or_graph: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/",
    name: str = DEFAULT_AGENT_NAME,
    description: str | None = None,
) -> None:
    """Load an agent (if given a spec string) and serve it over AG-UI.

    ``spec_or_graph`` is either an agent spec string (``module:attr`` or
    ``path/to/file.py:attr`` — resolved via the host layer's
    :func:`~langgraph_stream_parser.host.load_agent_spec`) or an already
    compiled graph. Blocks running a uvicorn server.
    """
    if isinstance(spec_or_graph, str):
        from ..host import load_agent_spec  # the host layer feeds AG-UI

        graph = load_agent_spec(spec_or_graph)
    else:
        graph = spec_or_graph
    app = build_app(graph, path=path, name=name, description=description)
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(_IMPORT_HINT) from e
    uvicorn.run(app, host=host, port=port)
