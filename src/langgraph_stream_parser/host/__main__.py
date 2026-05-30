"""``python -m langgraph_stream_parser.host`` — print the resolved shared config.

Shows each ``DEEPAGENT_*`` value, where it resolved from (default / TOML / env /
override), and the env var + ``deepagents.toml`` key that set it — so you never
have to remember the variable names. Hosts can ship their own subclass printer
for host-specific keys; this covers the shared core.
"""
from .config import HostConfig


def main() -> int:
    print(HostConfig.resolve().describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
