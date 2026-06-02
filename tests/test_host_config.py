"""Tests for the layered HostConfig resolver (defaults < TOML < env < overrides)."""
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from langgraph_stream_parser.host import HostConfig


@pytest.fixture
def isolated_global(tmp_path, monkeypatch):
    """Point the global deepagents config at an empty dir so the host machine's
    ~/.deepagents/config.toml can't leak into tests."""
    gdir = tmp_path / "global"
    gdir.mkdir()
    monkeypatch.setenv("DEEPAGENTS_CONFIG_HOME", str(gdir))
    return gdir


def _toml(dir_: Path, body: str) -> Path:
    p = dir_ / "deepagents.toml"
    p.write_text(body)
    return p


class TestResolveLayers:
    def test_defaults(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.port == 8050
        assert cfg.agent_spec is None
        assert cfg.workspace_root == Path(".")
        assert set(cfg.sources.values()) == {"default"}

    def test_env_layer(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(
            env={"DEEPAGENT_AGENT_SPEC": "a.py:g", "DEEPAGENT_PORT": "9000",
                 "DEEPAGENT_DEBUG": "true"},
            toml_start=tmp_path,
        )
        assert cfg.agent_spec == "a.py:g"
        assert cfg.port == 9000
        assert cfg.debug is True
        assert cfg.sources["agent_spec"] == "env:DEEPAGENT_AGENT_SPEC"

    def test_toml_layer(self, isolated_global, tmp_path):
        _toml(tmp_path, '[agent]\nspec = "x.py:graph"\n[server]\nport = 7000\n')
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.agent_spec == "x.py:graph"
        assert cfg.port == 7000
        assert cfg.sources["agent_spec"].startswith("toml")

    def test_precedence_toml_env_override(self, isolated_global, tmp_path):
        _toml(tmp_path, "[server]\nport = 1111\n")
        # toml only
        assert HostConfig.resolve(env={}, toml_start=tmp_path).port == 1111
        # env beats toml
        assert HostConfig.resolve(
            env={"DEEPAGENT_PORT": "2222"}, toml_start=tmp_path
        ).port == 2222
        # override beats env
        cfg = HostConfig.resolve(
            env={"DEEPAGENT_PORT": "2222"}, overrides={"port": 3333}, toml_start=tmp_path
        )
        assert cfg.port == 3333
        assert cfg.sources["port"] == "override"

    def test_none_override_ignored(self, isolated_global, tmp_path):
        cfg = HostConfig.resolve(
            env={"DEEPAGENT_PORT": "2222"}, overrides={"port": None}, toml_start=tmp_path
        )
        assert cfg.port == 2222

    def test_workspace_root_coerced_to_path(self, isolated_global, tmp_path):
        _toml(tmp_path, '[workspace]\nroot = "/tmp/ws"\n')
        cfg = HostConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.workspace_root == Path("/tmp/ws")


class TestIntrospection:
    def test_describe_lists_var_names_and_keys(self, isolated_global, tmp_path):
        text = HostConfig.resolve(env={}, toml_start=tmp_path).describe()
        assert "DEEPAGENT_AGENT_SPEC" in text   # the var you can never remember
        assert "agent.spec" in text             # its TOML key
        assert "[default]" in text

    def test_describe_marks_source(self, isolated_global, tmp_path):
        text = HostConfig.resolve(
            env={"DEEPAGENT_PORT": "9000"}, toml_start=tmp_path
        ).describe()
        assert "env:DEEPAGENT_PORT" in text


class TestSubclass:
    def test_subclass_adds_keys_to_same_chain(self, isolated_global, tmp_path):
        @dataclass
        class WebConfig(HostConfig):
            theme: str = "auto"
            _ENV: ClassVar[dict] = {"theme": ("DEEPAGENT_THEME", str)}
            _TOML: ClassVar[dict] = {"theme": "ui.theme"}

        _toml(tmp_path, '[ui]\ntheme = "solarized"\n')
        # toml provides theme...
        cfg = WebConfig.resolve(env={}, toml_start=tmp_path)
        assert cfg.theme == "solarized"
        # ...env overrides it, and base keys still resolve
        cfg = WebConfig.resolve(env={"DEEPAGENT_THEME": "dark"}, toml_start=tmp_path)
        assert cfg.theme == "dark"
        assert cfg.port == 8050
        assert cfg.sources["theme"] == "env:DEEPAGENT_THEME"
        assert "DEEPAGENT_THEME" in cfg.describe()


class TestFromEnvBackCompat:
    def test_from_env_skips_toml(self, isolated_global, tmp_path, monkeypatch):
        _toml(tmp_path, "[server]\nport = 1234\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DEEPAGENT_PORT", raising=False)
        # from_env ignores TOML even though deepagents.toml is in cwd
        assert HostConfig.from_env().port == 8050
