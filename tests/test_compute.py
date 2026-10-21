"""Launcher safeguards, with Modal mocked."""
import ast
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = ("compute/app.py", "compute/onboarding.py")


@pytest.fixture
def load(monkeypatch):
    monkeypatch.setenv("CS2243_PROJECT_ENV", "test-project")
    monkeypatch.setenv("CS2243_VOLUME_NAME", "staff-volume")
    monkeypatch.setenv("MODAL_ENVIRONMENT", "test-project")
    modal = MagicMock()
    modal.App.return_value.function.side_effect = lambda **kw: lambda f: f
    modal.App.return_value.local_entrypoint.side_effect = lambda **kw: lambda f: f
    monkeypatch.setitem(sys.modules, "modal", modal)

    def load(filename="compute/app.py"):
        spec = importlib.util.spec_from_file_location("app_under_test", ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    load.modal = modal
    return load


@pytest.fixture
def app(load, monkeypatch, tmp_path):
    app = load()
    app.LOG = tmp_path / "EXPERIMENT_LOG.md"
    app.LOG.write_text("- 2026-10-21 12:00 EST `base`: tests the no-op baseline\n")
    monkeypatch.setattr(app, "git_info", lambda: ("exp/noop", "abc123", False))
    monkeypatch.setattr(app, "run", MagicMock())
    return app


def launched(app):
    return list(app.run.map.call_args.args[0])


def test_uses_staff_volume(load):
    load()
    load.modal.Volume.from_name.assert_called_once_with("staff-volume")


@pytest.mark.parametrize("key", ["CS2243_PROJECT_ENV", "CS2243_VOLUME_NAME"])
def test_missing_staff_names_fail(load, monkeypatch, key):
    monkeypatch.setenv(key, "")
    with pytest.raises(ValueError, match="staff-assigned"):
        load()


def test_wrong_environment_rejected(load, monkeypatch):
    app = load()
    monkeypatch.setenv("MODAL_ENVIRONMENT", "main")
    with pytest.raises(RuntimeError, match="Wrong Modal Environment"):
        app.tasks("libero_spatial")


@pytest.mark.parametrize("filename", FILES)
def test_function_limits_and_gpus(filename):
    for node in ast.walk(ast.parse((ROOT / filename).read_text())):
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Call) and getattr(dec.func, "attr", None) == "function":
                kw = {k.arg: ast.literal_eval(k.value) for k in dec.keywords if k.arg in ("timeout", "max_containers", "gpu")}
                assert 0 < kw["timeout"] <= 24 * 3600 and 1 <= kw["max_containers"] <= 4
                assert kw.get("gpu") in (None, "T4", "L4")


def test_launch_requires_log_entry(app):
    with pytest.raises(SystemExit, match="EXPERIMENT_LOG.md"):
        app.main(name="unlogged")
    app.run.map.assert_not_called()


def test_launch_config_and_output_dir(app):
    app.main(name="base", overrides="train.lr=0.001")
    (cfg,) = launched(app)
    assert cfg["name"] == "base" and cfg["train"]["lr"] == 0.001
    assert cfg["run_dir"] == "/results/runs/exp/noop/base"
    assert cfg["git"] == {"branch": "exp/noop", "commit": "abc123", "dirty": False}


def test_sweep_expands_and_separates(app):
    app.main(name="base", overrides="train.lr=0.001,0.0001 train.epochs=0")
    cfgs = launched(app)
    assert [c["train"]["lr"] for c in cfgs] == [0.001, 0.0001]
    assert len({c["run_dir"] for c in cfgs}) == 2 and all(c["name"].startswith("base") for c in cfgs)
