"""Train the SAC Actor to imitate successful gamepad demonstrations."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from adaptive_manipulation.core.config import BCConfig, TrainingConfig
from adaptive_manipulation.core.paths import DEMONSTRATIONS_DIR, RUNS_DIR, TRAINING_CONFIG
from adaptive_manipulation.data.checkpoints import save_bc_checkpoint
from adaptive_manipulation.data.demonstrations import (
    atomic_json, environment_signature, load_demonstrations, split_episodes,
)
from adaptive_manipulation.learning.behavior_cloning import BCPolicy, initialize_std, imitation_metrics
from adaptive_manipulation.learning.evaluation import evaluate_policy, summarize
from adaptive_manipulation.simulation.gym_env import ManipulationEnv


def train_bc(demonstrations, output, environment_config, config=None):
    config = config or BCConfig()
    config.validate()
    environment_config.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('BC output directory is not empty; choose a new --output')
    device = config.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.device(device).type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    env = ManipulationEnv(**environment_config.environment_kwargs())
    try:
        signature = environment_signature(env)
        data, _ = load_demonstrations(demonstrations,success_only=True,expected_signature=signature)
        train_indices, validation_indices, split = split_episodes(data,config.validation_fraction,config.seed)
        # Record the exact files used and the fixed episode split for reproducibility.
        split['source'] = str(Path(demonstrations).resolve())
        split['source_hashes'] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(Path(demonstrations).glob('episode_[0-9][0-9][0-9][0-9].npz'))}
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/'config.json',dict(bc=asdict(config),environment=environment_config.to_dict(),device_used=device))
        atomic_json(output/'split.json',split)
        policy = BCPolicy(signature['observation_dim'],signature['action_dim'],environment_config.sac.hidden_sizes,device)
        initialize_std(policy.actor,config.initial_std)
        optimizer = torch.optim.Adam(policy.actor.parameters(),lr=config.learning_rate)
        def tensors(indices):
            return (torch.as_tensor(data['observations'][indices],device=device),
                    torch.as_tensor(data['actions'][indices],device=device))
        train_obs, train_actions = tensors(train_indices)
        val_obs, val_actions = tensors(validation_indices)
        print(f"BC device: {device} | {len(split['train_episode_ids'])} train / "
              f"{len(split['validation_episode_ids'])} validation scenarios | "
              f"{len(train_indices)} / {len(validation_indices)} transitions",flush=True)
        metric_names = ['mse','mae'] + [f'{name}_{i}' for name in ('mse','mae') for i in range(4)]
        fields = ['epoch'] + [f'{group}_{name}' for group in ('train','validation') for name in metric_names]
        best_loss, patience_reference, stale = float('inf'), float('inf'), 0
        best_epoch, completed_epoch, interrupted = 0, 0, False
        rng = np.random.default_rng(config.seed)
        with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as metrics_stream, \
                (output/'validation.csv').open('w',newline='',encoding='utf-8') as rollout_stream:
            metric_writer = csv.DictWriter(metrics_stream,fieldnames=fields)
            metric_writer.writeheader()
            rollout_writer = csv.DictWriter(rollout_stream,fieldnames=['epoch','mean_reward','success_rate','mean_distance'])
            rollout_writer.writeheader()
            def log_imitation(epoch):
                train_scores = imitation_metrics(policy.actor,train_obs,train_actions)
                validation_scores = imitation_metrics(policy.actor,val_obs,val_actions)
                row = dict(epoch=epoch,**{f'train_{k}':v for k,v in train_scores.items()},
                            **{f'validation_{k}':v for k,v in validation_scores.items()})
                if not all(math.isfinite(value) for value in row.values()):
                    raise FloatingPointError('Non-finite BC loss; training stopped')
                metric_writer.writerow(row)
                metrics_stream.flush()
                return row
            scores = log_imitation(0)
            best_loss = patience_reference = scores['validation_mse']
            save_bc_checkpoint(output/'best.pt',policy,environment_config,config,signature,split,0,scores)
            try:
                for epoch in range(1,config.epochs+1):
                    policy.actor.train()
                    order = rng.permutation(len(train_indices))
                    for start in range(0,len(order),config.batch_size):
                        indices = torch.as_tensor(order[start:start+config.batch_size],device=device)
                        mean, _ = policy.actor.distribution_parameters(train_obs[indices])
                        loss = F.mse_loss(torch.tanh(mean),train_actions[indices])
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite BC training loss')
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    policy.actor.eval()
                    scores = log_imitation(epoch)
                    completed_epoch = epoch
                    if scores['validation_mse'] < best_loss:
                        best_loss, best_epoch = scores['validation_mse'], epoch
                        save_bc_checkpoint(output/'best.pt',policy,environment_config,config,signature,split,epoch,scores)
                    if scores['validation_mse'] < patience_reference-config.min_delta:
                        patience_reference, stale = scores['validation_mse'], 0
                    else:
                        stale += 1
                    save_bc_checkpoint(output/'latest.pt',policy,environment_config,config,signature,split,epoch,scores)
                    print(f"Epoch {epoch}/{config.epochs} | train MSE {scores['train_mse']:.6f} | "
                          f"validation MSE {scores['validation_mse']:.6f} | best epoch {best_epoch}",flush=True)
                    if config.rollouts and config.evaluate_every and epoch % config.evaluate_every == 0:
                        rows = evaluate_policy(policy,env,split['validation_seeds'])
                        stats = summarize(rows)
                        rollout_writer.writerow(dict(epoch=epoch,**stats))
                        rollout_stream.flush()
                        print(f"MuJoCo validation | success {stats['success_rate']:.1%} | "
                              f"distance {stats['mean_distance']:.3f} m",flush=True)
                    if config.patience and stale >= config.patience:
                        print(f'Early stopping: no significant validation improvement for {stale} epochs',flush=True)
                        break
            except KeyboardInterrupt:
                interrupted = True
                save_bc_checkpoint(output/'latest.pt',policy,environment_config,config,signature,split,
                                   completed_epoch,scores,interrupted=True)
                print('Interrupted: current Actor saved in latest.pt; best.pt preserved.',flush=True)
        result = dict(best_epoch=best_epoch,best_validation_mse=best_loss,completed_epochs=completed_epoch,
                      interrupted=interrupted,split=split)
        if config.rollouts and not interrupted:
            from adaptive_manipulation.data.checkpoints import load_checkpoint
            best = BCPolicy.from_checkpoint(load_checkpoint(output/'best.pt'),device)
            rows = evaluate_policy(best,env,split['validation_seeds'])
            with (output/'best_validation_episodes.csv').open('w',newline='',encoding='utf-8') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result['best_validation_rollout'] = summarize(rows)
            print(f"Best Actor | held-out success {result['best_validation_rollout']['success_rate']:.1%}",flush=True)
        atomic_json(output/'summary.json',result)
        print(f'BC checkpoint: {output / "best.pt"}',flush=True)
        return result
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m adaptive_manipulation train-bc', description=__doc__)
    parser.add_argument('--demonstrations',default=str(DEMONSTRATIONS_DIR/'gamepad_30'))
    parser.add_argument('--output',default=str(RUNS_DIR/'bc_gamepad'))
    parser.add_argument('--config',default=str(TRAINING_CONFIG),help='Environment and Actor architecture configuration')
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--validation-fraction',type=float,default=.2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--patience',type=int,default=30,help='0 disables early stopping')
    parser.add_argument('--min-delta',type=float,default=1e-6)
    parser.add_argument('--initial-std',type=float,default=.3,help='Fixed pre-tanh Gaussian std, unused by deterministic imitation')
    parser.add_argument('--evaluate-every',type=int,default=20,help='MuJoCo validation interval; 0 disables periodic rollouts')
    parser.add_argument('--no-rollouts',action='store_true',help='Skip periodic and final MuJoCo evaluation')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--torch-threads',type=int,default=1)
    args = parser.parse_args(argv)
    environment_config = TrainingConfig.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8')))
    config = BCConfig(**{name:getattr(args,name) for name in BCConfig.__dataclass_fields__ if hasattr(args,name)})
    config.rollouts = not args.no_rollouts
    try:
        config.validate()
        train_bc(args.demonstrations,args.output,environment_config,config)
    except (ValueError,FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
