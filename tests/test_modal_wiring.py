"""Check local source packaging and launch safeguards without cloud execution."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

from src.sim import rollout

ROOT = Path(__file__).resolve().parents[1]


class ModalWiringTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "CS2243_PROJECT_ENV": "test-project",
            "CS2243_VOLUME_NAME": "staff-team-volume",
            "MODAL_ENVIRONMENT": "test-project",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.modal = MagicMock()
        app = self.modal.App.return_value
        app.function.side_effect = lambda **kwargs: lambda function: function
        app.local_entrypoint.side_effect = lambda **kwargs: lambda function: function
        self.modules = patch.dict(sys.modules, {"modal": self.modal})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def load_app(self, filename="modal_app.py"):
        spec = importlib.util.spec_from_file_location("app_under_test", ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_moved_simulator_is_importable_and_tasks_forward_to_it(self):
        app = self.load_app()
        with patch.object(rollout, "list_tasks", return_value=[{"task_id": 0}]) as listing:
            self.assertEqual(app.tasks("libero_spatial"), [{"task_id": 0}])
            listing.assert_called_once_with("libero_spatial")
        self.modal.Volume.from_name.assert_called_once_with("staff-team-volume")

    def test_wrong_environment_is_rejected_before_simulator_work(self):
        app = self.load_app()
        with patch.dict(os.environ, {"MODAL_ENVIRONMENT": "main"}):
            with patch.object(rollout, "list_tasks") as listing:
                with self.assertRaisesRegex(RuntimeError, "Wrong Modal Environment"):
                    app.tasks("libero_spatial")
                listing.assert_not_called()

    def test_missing_staff_configuration_fails_clearly(self):
        for key in ("CS2243_PROJECT_ENV", "CS2243_VOLUME_NAME"):
            with self.subTest(key=key), patch.dict(os.environ, {key: ""}):
                with self.assertRaisesRegex(ValueError, "staff-assigned"):
                    self.load_app()

    def test_every_remote_function_has_explicit_course_limits(self):
        for filename in ("modal_app.py", "modal_test.py"):
            tree = ast.parse((ROOT / filename).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                for decorator in node.decorator_list:
                    if (isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Attribute)
                            and decorator.func.attr == "function"):
                        values = {item.arg: item.value for item in decorator.keywords}
                        with self.subTest(file=filename, function=node.name):
                            self.assertGreater(ast.literal_eval(values["timeout"]), 0)
                            self.assertEqual(ast.literal_eval(values["max_containers"]), 1)


if __name__ == "__main__":
    unittest.main()
