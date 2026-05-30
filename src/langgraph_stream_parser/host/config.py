"""Shared ``DEEPAGENT_*`` configuration for hosts.

``HostConfig`` holds the keys every host has in common (agent spec, workspace
root, bind/title basics) and resolves them from one layered chain:

    defaults  <  deepagents.toml  <  DEEPAGENT_* env vars  <  explicit overrides

Host-specific keys (theme, auth, model, Jupyter token, ...) belong in each
host's own subclass — drift *below* the shared core is fine — but every host
gets the same resolution order, the same TOML files, and the same env-var
names, so there's one place to look.

Discoverability: ``HostConfig.resolve().describe()`` (or
``python -m langgraph_stream_parser.host``) prints each value, where it came
from, and the env var / TOML key that sets it — so you never have to remember
whether it's ``DEEPAGENT_SPEC`` or ``DEEPAGENT_AGENT_SPEC`` (it's the latter).
"""
import os
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, ClassVar

try:  # tomllib is stdlib on 3.11+; fall back to tomli; else the TOML layer is skipped.
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 path
    try:
        import tomli as _tomllib  # type: ignore
    except ModuleNotFoundError:
        _tomllib = None  # type: ignore

GLOBAL_TOML = Path.home() / ".deepagents" / "config.toml"
PROJECT_TOML = "deepagents.toml"


def _env_bool(value: str | None, default: bool = False) -> bool:
    """Parse an env-var string into a bool."""
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# ── TOML layer ───────────────────────────────────────────────────────


def _global_toml_path() -> Path:
    override = os.getenv("DEEPAGENTS_CONFIG_HOME")
    return (Path(override).expanduser() / "config.toml") if override else GLOBAL_TOML


def _find_project_toml(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (or cwd) looking for ``deepagents.toml``."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / PROJECT_TOML
        if candidate.is_file():
            return candidate
    return None


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_toml(path: Path) -> dict:
    if _tomllib is None:  # pragma: no cover
        return {}
    with path.open("rb") as f:
        return _tomllib.load(f)


def load_toml_config(start: Path | None = None) -> tuple[dict, list[Path]]:
    """Load + deep-merge the global and project ``deepagents.toml`` files.

    Global is ``~/.deepagents/config.toml`` (override the dir with
    ``DEEPAGENTS_CONFIG_HOME``); project is the nearest ``deepagents.toml`` at
    or above ``start``/cwd. Project wins on conflicts. Returns
    ``(merged_config, sources_used)``; ``({}, [])`` if no TOML reader is
    available (Python 3.10 without ``tomli``).
    """
    sources: list[Path] = []
    merged: dict = {}
    if _tomllib is None:  # pragma: no cover
        return merged, sources
    gpath = _global_toml_path()
    if gpath.is_file():
        merged = _deep_merge(merged, _read_toml(gpath))
        sources.append(gpath)
    ppath = _find_project_toml(start)
    if ppath is not None:
        merged = _deep_merge(merged, _read_toml(ppath))
        sources.append(ppath)
    return merged, sources


def _get_dotted(data: dict, dotted_key: str) -> Any:
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# ── Config dataclass ─────────────────────────────────────────────────


@dataclass
class HostConfig:
    """Shared configuration for deep-agent hosts.

    Subclass to add host-specific fields and extend the ``_ENV`` / ``_TOML``
    maps so they resolve through the same chain:

        @dataclass
        class WebConfig(HostConfig):
            theme: str = "auto"
            _ENV = {"theme": ("DEEPAGENT_THEME", str)}
            _TOML = {"theme": "ui.theme"}

    ``resolve()`` merges the maps across the MRO, so the subclass inherits all
    of ``HostConfig``'s keys and adds its own.
    """

    agent_spec: str | None = None     # DEEPAGENT_AGENT_SPEC ("path.py:var")
    workspace_root: Path = Path(".")  # DEEPAGENT_WORKSPACE_ROOT
    host: str = "localhost"           # DEEPAGENT_HOST
    port: int = 8050                  # DEEPAGENT_PORT
    debug: bool = False               # DEEPAGENT_DEBUG
    title: str = "Deep Agent"         # DEEPAGENT_TITLE

    # field -> (DEEPAGENT_* env var, caster). DEEPAGENT_AGENT_SPEC is canonical.
    _ENV: ClassVar[dict[str, tuple[str, Callable[[str], Any]]]] = {
        "agent_spec": ("DEEPAGENT_AGENT_SPEC", str),
        "workspace_root": ("DEEPAGENT_WORKSPACE_ROOT", Path),
        "host": ("DEEPAGENT_HOST", str),
        "port": ("DEEPAGENT_PORT", int),
        "debug": ("DEEPAGENT_DEBUG", _env_bool),
        "title": ("DEEPAGENT_TITLE", str),
    }
    # field -> dotted key in deepagents.toml
    _TOML: ClassVar[dict[str, str]] = {
        "agent_spec": "agent.spec",
        "workspace_root": "workspace.root",
        "host": "server.host",
        "port": "server.port",
        "debug": "debug",
        "title": "ui.title",
    }

    # ---- map collection across the subclass MRO ----

    @classmethod
    def _env_map(cls) -> dict[str, tuple[str, Callable[[str], Any]]]:
        merged: dict[str, tuple[str, Callable[[str], Any]]] = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "_ENV", {}))
        return merged

    @classmethod
    def _toml_map(cls) -> dict[str, str]:
        merged: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            merged.update(getattr(klass, "_TOML", {}))
        return merged

    # ---- resolution ----

    @classmethod
    def from_env(cls) -> "HostConfig":
        """Resolve from env vars + defaults only (no TOML, no overrides).

        Kept for back-compat; ``resolve()`` is the fuller entry point.
        """
        return cls.resolve(use_toml=False)

    @classmethod
    def resolve(
        cls,
        *,
        overrides: dict[str, Any] | None = None,
        toml_start: Path | None = None,
        env: dict[str, str] | None = None,
        use_toml: bool = True,
    ) -> "HostConfig":
        """Resolve config through ``defaults < TOML < env < overrides``.

        Each field's origin is recorded for ``describe()`` / ``sources``.

        Args:
            overrides: Highest-precedence values (e.g. CLI flags / Python args).
                ``None`` values are ignored so unset flags don't clobber.
            toml_start: Directory to start the ``deepagents.toml`` search from.
            env: Environment mapping (defaults to ``os.environ``).
            use_toml: Set False to skip the TOML layer entirely.
        """
        overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        env = os.environ if env is None else env
        toml_data, toml_paths = (load_toml_config(toml_start) if use_toml else ({}, []))
        env_map = cls._env_map()
        toml_map = cls._toml_map()

        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        for f in fields(cls):
            name = f.name
            if f.default is not MISSING:
                val: Any = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                val = f.default_factory()  # type: ignore[misc]
            else:
                val = None
            src = "default"

            tkey = toml_map.get(name)
            if tkey is not None:
                tv = _get_dotted(toml_data, tkey)
                if tv is not None:
                    val = _coerce(f, tv)
                    src = f"toml ({toml_paths[-1].name})" if toml_paths else "toml"

            if name in env_map:
                var, caster = env_map[name]
                ev = env.get(var)
                if ev is not None and ev != "":
                    val = caster(ev)
                    src = f"env:{var}"

            if name in overrides:
                val = overrides[name]
                src = "override"

            values[name] = val
            sources[name] = src

        obj = cls(**values)
        obj._sources = sources           # type: ignore[attr-defined]
        obj._toml_paths = toml_paths     # type: ignore[attr-defined]
        return obj

    def merge(self, **overrides: Any) -> "HostConfig":
        """Return a copy with non-``None`` overrides applied."""
        valid = {f.name for f in fields(self)}
        applied = {k: v for k, v in overrides.items() if v is not None and k in valid}
        return replace(self, **applied)

    # ---- introspection ----

    @property
    def sources(self) -> dict[str, str]:
        """Per-field origin from the last ``resolve()`` (field -> source)."""
        return getattr(self, "_sources", {})

    def describe(self) -> str:
        """Human-readable dump: value, source, and the env var / TOML key.

        This is what ``python -m langgraph_stream_parser.host`` prints.
        """
        env_map = type(self)._env_map()
        toml_map = type(self)._toml_map()
        src = self.sources
        lines = ["Resolved config  (value  [source]):", ""]
        for f in fields(self):
            value = getattr(self, f.name)
            origin = src.get(f.name, "default")
            hints = []
            if f.name in env_map:
                hints.append(f"env: {env_map[f.name][0]}")
            if f.name in toml_map:
                hints.append(f"toml: {toml_map[f.name]}")
            hint = f"   ({', '.join(hints)})" if hints else ""
            lines.append(f"  {f.name:<16} = {str(value):<26} [{origin}]{hint}")
        toml_paths = getattr(self, "_toml_paths", [])
        lines.append("")
        if toml_paths:
            lines.append("  TOML read from: " + ", ".join(str(p) for p in toml_paths))
        else:
            lines.append("  TOML: no deepagents.toml found")
        return "\n".join(lines)


def _coerce(f: Any, value: Any) -> Any:
    """Coerce a TOML value to the field's expected shape (Path fields only)."""
    if isinstance(getattr(f, "default", None), Path) and not isinstance(value, Path):
        return Path(value)
    return value
