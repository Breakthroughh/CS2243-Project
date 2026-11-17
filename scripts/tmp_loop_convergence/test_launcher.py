"""Check human log gate, unique output routing, and Modal resource limits without launches."""
import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

LAUNCHER = Path(__file__).parents[2] / "compute/analyze_loop_convergence.py"


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    modal = MagicMock()
    modal.App.return_value.function.side_effect = lambda **kw: lambda f: f
    modal.App.return_value.local_entrypoint.side_effect = lambda **kw: lambda f: f
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "compute.app", SimpleNamespace(
        PROJECT="test", ROOT=tmp_path, check_environment=lambda: None,
        git_info=lambda: ("exp/loop-convergence", "abc123", True), image=MagicMock(), volume=MagicMock()))
    spec = importlib.util.spec_from_file_location("convergence_launcher_under_test", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.analyze = MagicMock()
    (tmp_path / "EXPERIMENT_LOG.md").write_text("`synthetic-analysis`\n")
    return module


def test_unlogged_analysis_is_not_launched(launcher):
    with pytest.raises(SystemExit, match="EXPERIMENT_LOG.md"):
        launcher.main("unlogged")
    launcher.analyze.spawn.assert_not_called()


def test_logged_analysis_routes_to_its_own_branch_directory(launcher):
    launcher.main("synthetic-analysis")
    args = launcher.analyze.spawn.call_args.args
    assert args[:4] == ("/results/runs/exp/loop-convergence/synthetic-analysis", 16, 16, 0)
    assert args[4] == dict(branch="exp/loop-convergence", commit="abc123", dirty=True)
    launcher.analyze.spawn.return_value.get.assert_called_once_with()


def test_analysis_uses_one_bounded_t4():
    decorators = [d for n in ast.walk(ast.parse(LAUNCHER.read_text())) for d in getattr(n, "decorator_list", [])
                  if isinstance(d, ast.Call) and getattr(d.func, "attr", None) == "function"]
    assert len(decorators) == 1
    settings = {kw.arg: ast.literal_eval(kw.value) for kw in decorators[0].keywords
                if kw.arg in ("gpu", "timeout", "max_containers")}
    assert settings == dict(gpu="T4", timeout=1800, max_containers=1)


def test_runner_imports_from_only_the_files_shipped_in_the_image(tmp_path):
    # Import locally to collect image mounts, then re-import with only those files available.
    code = r'''
import importlib, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace

root, sandbox = map(Path, sys.argv[1:])
class Image:
    def __init__(self):
        self.mounts = []
    def __getattr__(self, name):
        return lambda *args, **kwargs: self
    def add_local_dir(self, local, remote, **kwargs):
        self.mounts.append((Path(local), remote))
        return self
class App:
    def __init__(self, *args, **kwargs):
        pass
    def function(self, **kwargs):
        return lambda function: function
    local_entrypoint = function
image = Image()
sys.modules["modal"] = SimpleNamespace(
    App=App, Image=image,
    Volume=SimpleNamespace(from_name=lambda *args: None),
    Secret=SimpleNamespace(from_name=lambda *args: None))
os.environ.update(CS2243_PROJECT_ENV="synthetic", CS2243_VOLUME_NAME="synthetic")
sys.path.insert(0, str(root))
launcher = importlib.import_module("compute.analyze_loop_convergence")
for local, remote in list(image.mounts):
    destination = sandbox / Path(remote).relative_to("/root")
    shutil.copytree(local, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
for name in list(sys.modules):
    if name == "compute" or name.startswith("compute."):
        del sys.modules[name]
sys.path = [str(sandbox)] + [p for p in sys.path if p != str(root)]
os.chdir(sandbox)
importlib.import_module("compute.analyze_loop_convergence")
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, str(LAUNCHER.parents[1]), str(tmp_path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
