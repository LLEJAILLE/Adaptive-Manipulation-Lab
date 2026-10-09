"""Versioned, atomic demonstration storage and success-filtered BC data loading."""

import json
import os
from pathlib import Path

import numpy as np

from adaptive_manipulation.data.checkpoints import scene_hash
from adaptive_manipulation.data.diagnostics import ACTION_FIELDS
from adaptive_manipulation.tasks.reach import MAX_DURATION
from adaptive_manipulation.core.contracts import OBSERVATION_VERSION, TASK_VERSION


def environment_signature(env):
    return {"observation_version": OBSERVATION_VERSION, "task_version": TASK_VERSION,
            "observation_dim": env.observation_space.shape[0], "action_dim": 4,
            "scene_hash": scene_hash(env.core.xml_path), "physics_steps": env.physics_steps,
            "max_steps": env.core.max_steps, "max_duration": MAX_DURATION,
            "decision_dt": env.decision_dt, "increments": env.increments.tolist(),
            "limits": env.limits.tolist(), "action_semantics": "incremental_position_v1"}


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def read_episode(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
        metadata = json.loads(str(archive['metadata'].item()))
    return metadata, data


class DemoRecorder:
    """Completed episode files are authoritative, including after a manifest crash."""

    def __init__(self, output, env, episodes=30, seed_start=3_000_000, resume=False):
        if episodes < 1 or seed_start < 0:
            raise ValueError('episodes must be positive and seed_start nonnegative')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.signature = environment_signature(env)
        self.seeds = list(range(seed_start, seed_start + episodes))
        self.manifest_path = self.output / 'session.json'
        self.completed = []
        self.rows = []
        self.current = None
        if resume:
            saved = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            if saved['signature'] != self.signature or saved['seeds'] != self.seeds:
                raise ValueError('Recording configuration, seeds or scene differ from this session')
            for path in sorted(self.output.glob('episode_[0-9][0-9][0-9][0-9].npz')):
                metadata, _ = read_episode(path)
                expected = len(self.completed)
                if (metadata['signature'] != self.signature or metadata['status'] != 'complete' or
                        expected >= len(self.seeds) or metadata['seed'] != self.seeds[expected] or
                        metadata['episode'] != expected + 1 or path.name != f'episode_{expected+1:04d}.npz'):
                    raise ValueError(f'Inconsistent demonstration episode: {path}')
                self.completed.append(metadata)
        elif any(self.output.iterdir()):
            raise ValueError('Demonstration directory is not empty; use --resume or a new output')
        self._write_manifest()

    @property
    def done(self):
        return len(self.completed) == len(self.seeds)

    @property
    def next_seed(self):
        if self.done:
            raise RuntimeError('All scenarios have already been recorded')
        return self.seeds[len(self.completed)]

    @property
    def successes(self):
        return sum(item['success'] for item in self.completed)

    def _write_manifest(self):
        atomic_json(self.manifest_path, dict(format_version=1, signature=self.signature,
                    seeds=self.seeds, completed=self.completed, diagnostic_fields=list(ACTION_FIELDS),
                    successes=self.successes, status='complete' if self.done else 'in_progress'))

    def begin(self, seed, spawn_position):
        if self.current is not None or seed != self.next_seed:
            raise ValueError('Start the next scheduled seed after finishing the active episode')
        self.current = dict(episode=len(self.completed)+1, seed=seed,
                            spawn_position=list(spawn_position), signature=self.signature)
        self.rows = []

    def add(self, observation, action, reward, next_observation, terminated, truncated, info):
        if self.current is None:
            raise RuntimeError('Call begin before recording a transition')
        obs = np.asarray(observation, dtype=np.float32).copy()
        next_obs = np.asarray(next_observation, dtype=np.float32).copy()
        act = np.asarray(action, dtype=np.float32).copy()
        if (obs.shape != (self.signature['observation_dim'],) or next_obs.shape != obs.shape or
                act.shape != (4,) or not all(np.isfinite(x).all() for x in (obs, next_obs, act)) or
                np.any(np.abs(act)>1) or not np.isfinite(reward)):
            raise ValueError('Invalid demonstration transition')
        if self.rows and not np.array_equal(self.rows[-1]['next_observations'], obs):
            raise ValueError('Non-contiguous demonstration observations')
        if self.rows and (self.rows[-1]['terminated'] or self.rows[-1]['truncated']):
            raise RuntimeError('Cannot append after the final transition')
        self.rows.append(dict(observations=obs, actions=act, rewards=float(reward),
            next_observations=next_obs, terminated=bool(terminated), truncated=bool(truncated),
            distances=float(info['distance']), elapsed=float(info['elapsed']),
            action_diagnostics=[info[key] for key in ACTION_FIELDS],
            reward_components=[info[key] for key in ('progress_reward','proximity_reward','time_penalty','terminal_reward')]))

    def finish(self, info):
        if not self.rows or not (self.rows[-1]['terminated'] or self.rows[-1]['truncated']):
            raise RuntimeError('Only completed scenarios can be counted as demonstrations')
        metadata = {**self.current, 'status':'complete', 'success':bool(info['success']),
                    'reason':info['reason'], 'steps':len(self.rows),
                    'total_reward':float(sum(row['rewards'] for row in self.rows)),
                    'final_distance':float(info['distance']), 'simulated_elapsed':float(info['elapsed'])}
        self._save(metadata, self.output / f"episode_{metadata['episode']:04d}.npz")
        self.completed.append(metadata)
        self.current, self.rows = None, []
        self._write_manifest()
        return metadata

    def interrupt(self):
        """Preserve partial attempts separately; they never enter default BC data."""
        if self.current is None:
            return
        if self.rows:
            metadata = {**self.current, 'status':'interrupted', 'success':False,
                        'reason':'user_interruption', 'steps':len(self.rows)}
            number = 1
            while True:
                path = self.output / f"partial_{metadata['episode']:04d}_{number:03d}.npz"
                if not path.exists():
                    break
                number += 1
            self._save(metadata, path)
        self.current, self.rows = None, []

    def _save(self, metadata, path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite demonstration: {path}')
        arrays = {key: np.asarray([row[key] for row in self.rows], dtype=np.float32)
                  for key in self.rows[0]}
        for key in ('rewards', 'terminated', 'truncated'):
            arrays[key] = arrays[key].reshape(-1,1)
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)
        os.replace(temporary, path)


def load_demonstrations(output, *, success_only=True, expected_signature=None):
    """Return transition arrays plus episode IDs for splitting BC by scenario."""
    folder = Path(output)
    manifest = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
    signature = manifest['signature']
    if (signature['observation_version'] != OBSERVATION_VERSION or signature['task_version'] != TASK_VERSION or
            expected_signature is not None and signature != expected_signature):
        raise ValueError('Demonstrations are incompatible with the current environment')
    batches, seeds = [], []
    for path in sorted(folder.glob('episode_[0-9][0-9][0-9][0-9].npz')):
        metadata, data = read_episode(path)
        if metadata['signature'] != signature or metadata['status'] != 'complete':
            raise ValueError(f'Invalid episode metadata: {path}')
        count = metadata['steps']
        if (count < 1 or data['observations'].shape != (count, signature['observation_dim']) or
                data['next_observations'].shape != data['observations'].shape or
                data['actions'].shape != (count, 4) or
                any(len(value) != count or not np.isfinite(value).all() for value in data.values()) or
                np.any(np.abs(data['actions'])>1) or
                not np.array_equal(data['next_observations'][:-1], data['observations'][1:]) or
                np.any(data['terminated'][:-1]) or np.any(data['truncated'][:-1]) or
                not (data['terminated'][-1,0] or data['truncated'][-1,0])):
            raise ValueError(f'Invalid or discontinuous transition data: {path}')
        if metadata['seed'] in seeds:
            raise ValueError('Demonstration seeds must be distinct')
        seeds.append(metadata['seed'])
        if success_only and not metadata['success']:
            continue
        data['episode_ids'] = np.full(count, metadata['episode'], dtype=np.int64)
        data['seeds'] = np.full(count, metadata['seed'], dtype=np.int64)
        batches.append(data)
    if not batches:
        raise ValueError('No completed successful demonstrations available' if success_only else 'No completed demonstrations available')
    return {key:np.concatenate([batch[key] for batch in batches]) for key in batches[0]}, signature


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog='python -m adaptive_manipulation inspect-demos', description='Inspect saved demonstrations (no training)')
    parser.add_argument('directory')
    args = parser.parse_args(argv)
    manifest = json.loads((Path(args.directory) / 'session.json').read_text(encoding='utf-8'))
    print(f"Scenarios: {len(manifest['completed'])}/{len(manifest['seeds'])}; successes: {manifest['successes']}")
    data, _ = load_demonstrations(args.directory, success_only=False)
    print(f"Validated {len(data['actions'])} transitions, observation dimension {data['observations'].shape[1]}")


def split_episodes(data, validation_fraction=.2, seed=42):
    """Never put neighboring transitions from a scenario in both partitions."""
    episodes = np.unique(data['episode_ids'])
    if len(episodes) < 2:
        raise ValueError('At least two successful scenarios are needed for training and validation')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    shuffled = np.random.default_rng(seed).permutation(episodes)
    count = min(len(episodes)-1, max(1, int(round(len(episodes)*validation_fraction))))
    validation_ids, train_ids = sorted(shuffled[:count].tolist()), sorted(shuffled[count:].tolist())
    train_indices = np.flatnonzero(np.isin(data['episode_ids'], train_ids))
    validation_indices = np.flatnonzero(np.isin(data['episode_ids'], validation_ids))
    def seeds_for(indices):
        return sorted(np.unique(data['seeds'][indices]).tolist())
    split = dict(train_episode_ids=train_ids, validation_episode_ids=validation_ids,
                 train_seeds=seeds_for(train_indices), validation_seeds=seeds_for(validation_indices),
                 train_transitions=len(train_indices), validation_transitions=len(validation_indices))
    if set(split['train_seeds']) & set(split['validation_seeds']):
        raise ValueError('Training and validation seeds overlap')
    return train_indices, validation_indices, split


if __name__ == '__main__':
    main()
