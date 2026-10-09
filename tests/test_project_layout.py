"""Regression checks for CLI dispatch, resources and pre-move checkpoints."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from adaptive_manipulation import cli
from adaptive_manipulation.data.checkpoints import load_checkpoint
from adaptive_manipulation.simulation.environment import ScenarioResult
from adaptive_manipulation.core.config import TrainingConfig
from adaptive_manipulation.workflows.train_sac import train


class ProjectLayoutTests(unittest.TestCase):
    def test_new_sac_run_preserves_existing_files_without_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'preserved checkpoint')
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'not empty'):
                    train(TrainingConfig(device='cpu'), directory)
            self.assertEqual(checkpoint.read_bytes(), b'preserved checkpoint')
            self.assertEqual(list(Path(directory).iterdir()), [checkpoint])

    def test_dispatch_preserves_command_arguments(self):
        command = types.SimpleNamespace(main=lambda args: args)
        with patch.object(cli, 'import_module', return_value=command) as importer:
            result = cli.main(['train-sac', '--resume', 'run/latest.pt', '--steps', '500'])
        importer.assert_called_once_with('adaptive_manipulation.workflows.train_sac')
        self.assertEqual(result, ['--resume', 'run/latest.pt', '--steps', '500'])

    def test_general_help_does_not_import_simulation_or_training(self):
        root = Path(__file__).resolve().parents[1]
        code = """
import sys
from adaptive_manipulation.cli import main
try:
    main(['--help'])
except SystemExit as error:
    assert error.code == 0
assert not {'torch', 'mujoco', 'pygame', 'tkinter'} & set(sys.modules)
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('train-bc', result.stdout)

    def test_resources_work_outside_project_directory(self):
        root = Path(__file__).resolve().parents[1]
        code = f"""
import sys
sys.path.insert(0, {str(root)!r})
from adaptive_manipulation.core.paths import MODEL_PATH, TRAINING_CONFIG
from adaptive_manipulation.simulation.gym_env import ManipulationEnv
assert MODEL_PATH.is_file() and TRAINING_CONFIG.is_file()
env = ManipulationEnv(render_mode='machine')
try:
    obs, _ = env.reset(seed=42)
    assert obs.shape == (38,)
finally:
    env.close()
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', code], cwd=directory,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pre_reorganization_scenario_is_deserialized(self):
        result = ScenarioResult(1, True, 'contact', 1., (0., 0., 0.),
                                200., 201., 50, .05, 42)
        old_module = types.ModuleType('environment')
        old_module.ScenarioResult = ScenarioResult
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            original_module = ScenarioResult.__module__
            try:
                ScenarioResult.__module__ = 'environment'
                with patch.dict(sys.modules, {'environment': old_module}):
                    torch.save({'version': 2, 'result': result}, path)
            finally:
                ScenarioResult.__module__ = original_module
            payload = load_checkpoint(path)
        self.assertEqual(payload['result'], result)
        self.assertIs(type(payload['result']), ScenarioResult)


if __name__ == '__main__':
    unittest.main()
