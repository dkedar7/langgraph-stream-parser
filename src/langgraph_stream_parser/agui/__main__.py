"""``langstage-agui`` — serve a LangGraph agent over the AG-UI protocol.

    langstage-agui --agent my_agent.py:graph
    langstage-agui --demo                       # keyless echo agent, no API key
    langstage-agui --agent langstage_hermes.agent:graph --port 9000

The agent spec resolves through the shared host config chain, so
``LANGSTAGE_AGENT_SPEC`` / ``langstage.toml`` work too (legacy ``DEEPAGENT_*``
still honoured).
"""
from __future__ import annotations

import sys

DEMO_SPEC = "langgraph_stream_parser.demo.stub:graph"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="langstage-agui",
        description="Serve a LangGraph agent over the AG-UI protocol.",
    )
    parser.add_argument(
        "--agent",
        "-a",
        dest="agent",
        default=None,
        help="Agent spec (module:attr or path/to/file.py:attr). "
        "Falls back to LANGSTAGE_AGENT_SPEC / langstage.toml.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Serve the built-in keyless demo agent (no API key needed).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000).")
    parser.add_argument("--path", default="/", help="Endpoint path (default '/').")
    parser.add_argument("--name", default=None, help="Agent display name for AG-UI clients.")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved host config and exit.",
    )
    args = parser.parse_args(argv)

    from ..host import HostConfig

    if args.show_config:
        print(HostConfig.resolve().describe())
        return 0

    if args.demo and args.agent:
        print("error: --demo and --agent are mutually exclusive", file=sys.stderr)
        return 2

    if args.demo:
        spec: str | None = DEMO_SPEC
    else:
        cfg = HostConfig.resolve(overrides={"agent_spec": args.agent})
        spec = cfg.agent_spec

    if not spec:
        print(
            "error: no agent spec — pass --agent, --demo, set LANGSTAGE_AGENT_SPEC, "
            "or add [agent].spec to langstage.toml",
            file=sys.stderr,
        )
        return 2

    from . import DEFAULT_AGENT_NAME, serve

    name = args.name or ("Demo Agent" if args.demo else DEFAULT_AGENT_NAME)
    print(f"Serving {spec!r} over AG-UI at http://{args.host}:{args.port}{args.path}")
    serve(spec, host=args.host, port=args.port, path=args.path, name=name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
