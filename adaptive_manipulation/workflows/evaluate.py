"""Deterministic validation and held-out evaluation, optionally with a viewer."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch

from adaptive_manipulation.data.checkpoints import check_environment_compatibility, load_checkpoint, scene_hash
from adaptive_manipulation.core.config import TrainingConfig
from adaptive_manipulation.core.paths import RUNS_DIR
from adaptive_manipulation.learning.evaluation import evaluate_policy, summarize
from adaptive_manipulation.simulation.gym_env import ManipulationEnv
from adaptive_manipulation.learning.sac import SACAgent


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m adaptive_manipulation evaluate', description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument('--split',choices=('new','train','validation'),default='new',
                        help='BC: replay a saved scenario split, or evaluate unseen seeds (default)')
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", help="Output CSV (default: a new timestamped file in runs/evaluations/)")
    args = parser.parse_args(argv)
    payload = load_checkpoint(args.checkpoint)
    config = TrainingConfig.from_dict(payload["config"])
    is_bc = payload.get('kind') == 'bc'
    torch.set_num_threads(payload['bc_config']['torch_threads'] if is_bc else config.torch_threads)
    if scene_hash(config.xml_path) != payload["scene_hash"]:
        parser.error("Scene XML differs from checkpoint")
    if (args.episodes is not None and args.episodes < 1 or
            args.seed_start is not None and args.seed_start < 0):
        parser.error("episodes must be positive and seed-start nonnegative")
    if args.split != 'new':
        if not is_bc:
            parser.error('--split train/validation is available only for BC checkpoints')
        if args.seed_start is not None:
            parser.error('Do not use --seed-start with a saved BC split')
        available = payload['split'][f'{args.split}_seeds']
        seeds = args.seeds or available[:args.episodes]
        if not set(seeds).issubset(available):
            parser.error('Requested seeds must belong to the selected BC split')
    else:
        seed_start = args.seed_start if args.seed_start is not None else 2_000_000
        episodes = args.episodes if args.episodes is not None else 100
        seeds = args.seeds or list(range(seed_start,seed_start+episodes))
    if len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        parser.error("Use distinct nonnegative seeds")
    for seed in seeds:
        if is_bc:
            used = set(payload['split']['train_seeds'] + payload['split']['validation_seeds'])
            if args.split == 'new' and seed in used:
                parser.error('New BC evaluation seeds must be outside both saved demonstration splits')
        elif (config.is_training_seed(seed) or config.is_validation_seed(seed) or seed in
                (payload.get('training',{}).get('bc_initialization') or {}).get('reserved_validation_seeds',[])):
            parser.error("Evaluation seeds must be outside training AND model-selection validation pools")
    env = ManipulationEnv(render_mode="human" if args.render else "machine", **config.environment_kwargs())
    try:
        check_environment_compatibility(payload, env)
        if is_bc:
            from adaptive_manipulation.learning.behavior_cloning import BCPolicy
            agent = BCPolicy.from_checkpoint(payload,args.device)
        else:
            agent = SACAgent(payload["agent"]["observation_dim"], payload["agent"]["action_dim"], config.sac, args.device)
            agent.load_state_dict(payload["agent"])
        rows = evaluate_policy(agent, env, seeds)
    finally:
        env.close()
    if not rows:
        print("No completed evaluation episodes")
        return
    path = Path(args.csv) if args.csv else RUNS_DIR / 'evaluations' / f"{Path(args.checkpoint).parent.name}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"Seed {row['seed']} | Steps {row['steps']} | Reward {row['total_reward']:.2f} | "
              f"Success {row['success']} | Distance {row['final_distance']:.4f} m")
    stats = summarize(rows)
    print(f"Evaluation CSV: {path}")
    print(f"Evaluated {len(rows)} scenarios | Mean reward {stats['mean_reward']:.2f} | "
          f"Success rate {stats['success_rate']:.1%} | Mean distance {stats['mean_distance']:.4f} m")


if __name__ == "__main__":
    main()
