"""Validate a BC transfer and import only training demonstrations into SAC replay."""

import hashlib
from pathlib import Path

import numpy as np

from adaptive_manipulation.data.checkpoints import check_environment_compatibility, load_checkpoint
from adaptive_manipulation.data.demonstrations import environment_signature, load_demonstrations


def prepare_bc_initialization(checkpoint,config,env,replay,demonstrations=None):
    path = Path(checkpoint).resolve()
    payload = load_checkpoint(path)
    if payload.get('kind') != 'bc':
        raise ValueError('--init-bc requires an Actor-only BC checkpoint')
    check_environment_compatibility(payload,env)
    if (tuple(payload['hidden_sizes']) != tuple(config.sac.hidden_sizes) or
            payload['action_dim'] != env.action_space.shape[0]):
        raise ValueError('BC Actor architecture differs from SAC; use the BC architecture')
    split = payload['split']
    train_ids, validation_ids = set(split['train_episode_ids']),set(split['validation_episode_ids'])
    train_seeds, validation_seeds = set(split['train_seeds']),set(split['validation_seeds'])
    if not train_ids or not validation_ids or train_ids & validation_ids or train_seeds & validation_seeds:
        raise ValueError('Invalid or overlapping BC train/validation split')
    source = Path(demonstrations or split['source']).resolve()
    data,_ = load_demonstrations(source,success_only=True,expected_signature=environment_signature(env))
    mask = np.isin(data['episode_ids'],list(train_ids))
    if (set(data['episode_ids'][mask].tolist()) != train_ids or
            set(data['seeds'][mask].tolist()) != train_seeds or
            np.any(np.isin(data['seeds'][mask],list(validation_seeds)))):
        raise ValueError('Demonstrations do not match the BC training split')
    count = int(mask.sum())
    if count != split['train_transitions']:
        raise ValueError('BC training demonstration count changed')
    if replay.capacity < count or count < config.batch_size:
        raise ValueError('Replay capacity must hold all BC training transitions, and batch_size must not exceed their count')
    for episode in train_ids:
        filename = f'episode_{episode:04d}.npz'
        expected = split.get('source_hashes',{}).get(filename)
        if expected is None or hashlib.sha256((source/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'BC training demonstration changed since cloning: {filename}')
    # Use the original split by default; configurable explicit lists may extend
    # training, but may never contain the BC held-out seeds.
    if config.train_seeds is None:
        config.train_seeds = sorted(train_seeds)
    if config.validation_seeds is None:
        config.validation_seeds = sorted(validation_seeds)
    if any(config.is_training_seed(s) for s in validation_seeds):
        raise ValueError('BC held-out validation seeds cannot be used for SAC training')
    if any(config.is_validation_seed(s) for s in train_seeds):
        raise ValueError('SAC validation cannot contain BC training demonstration seeds')
    config.start_steps = config.update_after = 0
    config.validate()
    for index in np.flatnonzero(mask):
        replay.add(*(data[key][index] for key in ('observations','actions','rewards','next_observations','terminated','truncated')))
    metadata = dict(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    bc_epoch=payload['epoch'],demonstrations=str(source),
                    demo_transitions=count,demo_episode_ids=sorted(train_ids),demo_seeds=sorted(train_seeds),
                    reserved_validation_seeds=sorted(validation_seeds),
                    critic_warmup_updates=config.bc_critic_warmup_updates)
    return payload,metadata
