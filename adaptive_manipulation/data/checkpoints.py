"""Atomic local checkpoints, including simulation and random states."""

import hashlib
from dataclasses import asdict
import os
from pathlib import Path
import pickle
import random
import types

import numpy as np
import torch

from adaptive_manipulation.core.contracts import OBSERVATION_VERSION, TASK_VERSION
from adaptive_manipulation.core.paths import MODEL_PATH


def scene_hash(xml_path=None):
    path = Path(xml_path) if xml_path else MODEL_PATH
    return hashlib.sha256(path.read_bytes()).hexdigest()


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path, agent, config, training, replay=None, environment=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 2, "observation_version": OBSERVATION_VERSION,
               "task_version": TASK_VERSION, "agent": agent.state_dict(), "config": config.to_dict(),
               "scene_hash": scene_hash(config.xml_path), "training": training,
               "random": random_state(), "replay": replay.state_dict() if replay is not None else None,
               "environment": environment.state_dict() if environment is not None else None}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Full local training checkpoints contain NumPy and MuJoCo scenario objects.
    # Only load checkpoints produced by this project from a trusted source.
    payload = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_legacy_pickle)
    if payload.get("version") not in (1, 2):
        raise ValueError("Unsupported checkpoint version")
    return payload


def check_environment_compatibility(payload, env):
    observation_dim = (payload['observation_dim'] if payload.get('kind') == 'bc'
                       else payload['agent']['observation_dim'])
    if (payload.get("observation_version", 1) != OBSERVATION_VERSION or
            payload.get("task_version", 1) != TASK_VERSION or
            observation_dim != env.observation_space.shape[0]):
        raise ValueError("Checkpoint incompatible with the corrected task: remaining time/steps "
                         "are now observed and deadline failures are terminal. Keep old runs for "
                         "comparison; start a new model with the current demonstration format.")
    if payload.get('kind') == 'bc':
        from adaptive_manipulation.data.demonstrations import environment_signature
        if payload['signature'] != environment_signature(env):
            raise ValueError('BC checkpoint control settings or scene differ from the environment')


class _LegacyUnpickler(pickle.Unpickler):
    """Read pre-reorganization ScenarioResult objects without root-level shims."""
    def find_class(self,module,name):
        if module == 'environment' and name == 'ScenarioResult':
            module = 'adaptive_manipulation.simulation.environment'
        return super().find_class(module,name)


_legacy_pickle = types.ModuleType('adaptive_manipulation_legacy_pickle')
_legacy_pickle.Unpickler = _LegacyUnpickler
_legacy_pickle.load = pickle.load
_legacy_pickle.loads = pickle.loads


def save_bc_checkpoint(path, policy, environment_config, bc_config, signature, split, epoch, metrics, interrupted=False):
    payload = dict(version=2,kind='bc',observation_version=OBSERVATION_VERSION,task_version=TASK_VERSION,
                   observation_dim=policy.observation_dim,action_dim=policy.action_dim,
                   hidden_sizes=list(environment_config.sac.hidden_sizes),
                   actor={key:value.detach().cpu() for key,value in policy.actor.state_dict().items()},
                   config=environment_config.to_dict(),bc_config=asdict(bc_config),signature=signature,
                   scene_hash=signature['scene_hash'],split=split,epoch=epoch,metrics=metrics,
                   interrupted=interrupted)
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)
