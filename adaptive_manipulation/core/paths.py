"""Default checkout resources; explicit CLI paths remain relative to the caller."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / 'models' / 'robot_arm.xml'
TRAINING_CONFIG = PROJECT_ROOT / 'configs' / 'training.json'
RUNS_DIR = PROJECT_ROOT / 'runs'
DEMONSTRATIONS_DIR = PROJECT_ROOT / 'demonstrations'
