"""`python -m langgraph_stream_parser.host` argparse (gh #-dogfood).

It used to ignore all args — including --help, which was a silent no-op. It now
has a tiny argparse so -h/--help works and unknown flags error.
"""
import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "langgraph_stream_parser.host", *args],
        capture_output=True,
        text=True,
    )


def test_help_shows_usage_and_exits_zero():
    r = _run("--help")
    assert r.returncode == 0
    assert "usage:" in r.stdout.lower()


def test_unknown_flag_errors():
    r = _run("--bogus")
    assert r.returncode != 0
    assert "unrecognized arguments" in r.stderr.lower() or "error" in r.stderr.lower()


def test_no_args_prints_config():
    r = _run()
    assert r.returncode == 0
    # describe() output includes the canonical env-var hints.
    assert "LANGSTAGE_" in r.stdout
