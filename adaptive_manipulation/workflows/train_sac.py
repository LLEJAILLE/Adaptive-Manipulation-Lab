"""Headless SAC training with CSV metrics, validation, and resumable checkpoints."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from adaptive_manipulation.data.checkpoints import check_environment_compatibility, load_checkpoint, restore_random_state, save_checkpoint, scene_hash
from adaptive_manipulation.core.config import TrainingConfig
from adaptive_manipulation.data.logging import CSVLog
from adaptive_manipulation.core.paths import RUNS_DIR
from adaptive_manipulation.data.diagnostics import ACTION_FIELDS, CRITIC_FIELDS, aggregate_critics
from adaptive_manipulation.learning.evaluation import evaluate_policy, summarize
from adaptive_manipulation.learning.replay_buffer import ReplayBuffer
from adaptive_manipulation.simulation.gym_env import ManipulationEnv
from adaptive_manipulation.learning.sac import SACAgent




def train(config, output, resume=None, init_bc=None, demonstrations=None):
    if resume is not None and init_bc is not None:
        raise ValueError('--resume and --init-bc are mutually exclusive')
    if demonstrations is not None and init_bc is None:
        raise ValueError('--demonstrations is used only with --init-bc')
    config.validate()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. Install GPU dependencies with "
            "python -m pip install -r requirements-cuda.txt, or use --device cpu.")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(f"Training device: {device} ({device_name})", flush=True)
    output = Path(output)
    if resume is None and output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use --resume or a new --output")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.torch_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    env = ManipulationEnv(**config.environment_kwargs())
    validation_env = ManipulationEnv(**config.environment_kwargs())
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], config.sac, config.device)
    replay = ReplayBuffer(config.replay_capacity, agent.observation_dim, agent.action_dim, config.seed)
    seed_rng = np.random.default_rng(config.seed)
    env.action_space.seed(config.seed)
    state = {"global_step": 0, "episodes": 0, "successes": [], "rewards": [],
             "best_success_rate": -1., "best_mean_reward": -float("inf"),
             "elapsed": 0., "losses": [], "action_totals": dict.fromkeys(ACTION_FIELDS, 0.),
             "action_samples": 0, "bc_initialization": None}
    if resume is not None:
        try:
            if resume["scene_hash"] != scene_hash(config.xml_path):
                raise ValueError("Scene XML differs from checkpoint")
            check_environment_compatibility(resume, env)
            if resume["replay"] is None or resume["environment"] is None:
                raise ValueError("Resume requires a full training checkpoint (latest.pt or step_*.pt)")
        except Exception:
            env.close()
            validation_env.close()
            raise
        agent.load_state_dict(resume["agent"])
        replay.load_state_dict(resume["replay"])
        env.load_state_dict(resume["environment"])
        state = resume["training"]
        seed_rng.bit_generator.state = state["seed_rng"]
        restore_random_state(resume["random"])
        observation = env._observation()
    else:
        if init_bc is not None:
            try:
                from adaptive_manipulation.workflows.bc_transfer import prepare_bc_initialization
                bc_payload,metadata = prepare_bc_initialization(init_bc,config,env,replay,demonstrations)
                agent.actor.load_state_dict(bc_payload['actor'])
                state['bc_initialization'] = metadata
            except Exception:
                env.close()
                validation_env.close()
                raise
            print(f"BC Actor loaded | {metadata['demo_transitions']} training demonstration transitions | "
                  f"{config.bc_critic_warmup_updates} critic-only updates",flush=True)
        observation, _ = env.reset(seed=config.sample_training_seed(seed_rng))
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    if state.get('bc_initialization') is not None:
        (output/'bc_initialization.json').write_text(json.dumps(state['bc_initialization'],indent=2),encoding='utf-8')
    step_at_start = state["global_step"]
    session_start = time.perf_counter()
    previous_elapsed = state["elapsed"]
    resume_step = step_at_start if resume is not None else None
    episode_log = CSVLog(output / "episodes.csv", ["global_step", "episode", "seed", "steps", "total_reward",
                         "success", "final_distance", "success_rate_100", "elapsed", "simulated_elapsed", "reason"], resume_step)
    stats_log = CSVLog(output / "stats.csv", ["global_step", "episodes", "mean_reward_100", "success_rate_100",
                       "actor_loss", "critic_loss", "alpha_loss", "alpha", "updates", "steps_per_second", "elapsed",
                       "phase", "actor_update_fraction", "critic_warmup_remaining"], resume_step)
    validation_log = CSVLog(output / "validation.csv", ["global_step", "episodes", "mean_reward", "success_rate", "mean_distance"], resume_step)
    critic_log = CSVLog(output / "critics.csv", ["global_step", "updates", *CRITIC_FIELDS], resume_step)
    action_log = CSVLog(output / "actions.csv", ["global_step", "samples", *ACTION_FIELDS], resume_step)

    def elapsed():
        return previous_elapsed + time.perf_counter() - session_start

    def snapshot():
        return {**state, "seed_rng": seed_rng.bit_generator.state, "elapsed": elapsed()}

    def save(name, full=True):
        save_checkpoint(output / name, agent, config, snapshot(),
                        replay=replay if full else None, environment=env if full else None)

    def validate():
        rows = evaluate_policy(agent, validation_env,config.validation_seed_pool())
        stats = summarize(rows)
        validation_log.write({"global_step": state["global_step"], "episodes": state["episodes"], **stats})
        print(f"Validation | Step {state['global_step']} | Mean reward {stats['mean_reward']:.2f} | "
              f"Success rate {stats['success_rate']:.1%}", flush=True)
        score = (stats["success_rate"], stats["mean_reward"])
        if score > (state["best_success_rate"], state["best_mean_reward"]):
            state["best_success_rate"], state["best_mean_reward"] = score
            save("best.pt", full=False)

    def log_statistics():
        losses = state["losses"]
        means = {k: float(np.mean([x[k] for x in losses])) if losses else ""
                 for k in ("actor_loss", "critic_loss", "alpha_loss")}
        seconds = time.perf_counter() - session_start
        rate = (state["global_step"] - step_at_start) / max(seconds, 1e-9)
        mean_reward = float(np.mean(state["rewards"])) if state["rewards"] else 0.
        success_rate = float(np.mean(state["successes"])) if state["successes"] else 0.
        stats_log.write({"global_step": state["global_step"], "episodes": state["episodes"],
                         "mean_reward_100": mean_reward, "success_rate_100": success_rate,
                         **means, "alpha": float(agent.alpha), "updates": agent.updates,
                         "steps_per_second": rate, "elapsed": elapsed(),
                         "phase": "critic_warmup" if critic_warmup() else "sac",
                         "actor_update_fraction": float(np.mean([x.get('actor_updated',1.) for x in losses])) if losses else "",
                         "critic_warmup_remaining": max(0,config.bc_critic_warmup_updates-agent.updates) if state.get('bc_initialization') else 0})
        critic_log.write({"global_step": state["global_step"], "updates": agent.updates,
                          **aggregate_critics(losses)})
        samples = state["action_samples"]
        action_log.write({"global_step": state["global_step"], "samples": samples,
                          **{k: v / samples if samples else "" for k, v in state["action_totals"].items()}})
        print(f"Stats | Step {state['global_step']} | Mean reward {mean_reward:.2f} | "
              f"Success rate {success_rate:.1%} | Actor {means['actor_loss']} | "
              f"Critic {means['critic_loss']} | {rate:.1f} steps/s", flush=True)
        state["losses"] = []
        state["action_totals"] = dict.fromkeys(ACTION_FIELDS, 0.)
        state["action_samples"] = 0

    def critic_warmup():
        return state.get('bc_initialization') is not None and agent.updates < config.bc_critic_warmup_updates

    print("Episode | Seed | Steps | Total Reward | Success | Final Distance | Success Rate (100) | Elapsed Time", flush=True)
    try:
        if init_bc is not None:
            save('initialized.pt')
        if state.get('bc_initialization') and state['global_step'] == 0 and state['best_success_rate'] < 0:
            validate()  # Baseline protects best.pt if SAC subsequently regresses.
        while state["global_step"] < config.total_steps and (
                config.max_episodes is None or state["episodes"] < config.max_episodes):
            action = (env.action_space.sample() if state["global_step"] < config.start_steps
                      else agent.act(observation,deterministic=critic_warmup()))
            next_observation, reward, terminated, truncated, info = env.step(action)
            for key in ACTION_FIELDS:
                state["action_totals"][key] += info[key]
            state["action_samples"] += 1
            replay.add(observation, action, reward, next_observation, terminated, truncated)
            observation = next_observation
            state["global_step"] += 1
            if state["global_step"] >= config.update_after and len(replay) >= config.batch_size:
                for _ in range(config.updates_per_step):
                    state["losses"].append(agent.update(replay.sample(config.batch_size, config.device),
                                                       update_actor=not critic_warmup()))
            if terminated or truncated:
                state["episodes"] += 1
                state["successes"] = (state["successes"] + [bool(info["success"])])[-100:]
                state["rewards"] = (state["rewards"] + [info["total_reward"]])[-100:]
                success_rate = float(np.mean(state["successes"]))
                row = {key: info[key] for key in ("seed", "steps", "total_reward", "success", "final_distance", "reason")}
                episode_log.write({**row, "global_step": state["global_step"], "episode": state["episodes"],
                                   "success_rate_100": success_rate, "elapsed": elapsed(), "simulated_elapsed": info["elapsed"]})
                print(f"{state['episodes']} | {info['seed']} | {info['steps']} | {info['total_reward']:.2f} | "
                      f"{info['success']} | {info['final_distance']:.4f} m | {success_rate:.1%} | {elapsed():.1f} s", flush=True)
                observation, _ = env.reset(seed=config.sample_training_seed(seed_rng))
            if state["global_step"] % config.log_every == 0:
                log_statistics()
            if state["global_step"] % config.evaluate_every == 0:
                validate()
            if state["global_step"] % config.checkpoint_every == 0:
                save(f"step_{state['global_step']:09d}.pt")
                save("latest.pt")
        if state["global_step"] % config.log_every:
            log_statistics()
        if state["global_step"] % config.evaluate_every:
            validate()
    except KeyboardInterrupt:
        print("Interrupted; saving current training state.", flush=True)
    finally:
        try:
            save("latest.pt")
        finally:
            for log in (episode_log, stats_log, validation_log, critic_log, action_log):
                log.close()
            env.close()
            validation_env.close()
    print(f"Checkpoint: {output / 'latest.pt'}", flush=True)
    return snapshot()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m adaptive_manipulation train-sac', description=__doc__)
    parser.add_argument("--config", help="JSON configuration (see configs/training.json)")
    parser.add_argument("--resume", help="Full training checkpoint")
    parser.add_argument('--init-bc',help='Initialize a new SAC run from a BC Actor checkpoint')
    parser.add_argument('--demonstrations',help='Override the demonstration directory saved in the BC checkpoint')
    parser.add_argument('--critic-warmup-updates',type=int,help='BC transfer: critic-only updates before Actor/alpha learning (default 1000)')
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int, help="Total step budget, including restored steps")
    parser.add_argument("--episodes", type=int, help="Total completed episode budget")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    if args.resume and args.config:
        parser.error("--resume uses the saved configuration; omit --config")
    if args.resume and (args.init_bc or args.demonstrations or args.critic_warmup_updates is not None):
        parser.error('--resume cannot be combined with BC initialization options')
    if not args.init_bc and (args.demonstrations or args.critic_warmup_updates is not None):
        parser.error('--demonstrations and --critic-warmup-updates require --init-bc')
    resume = load_checkpoint(args.resume) if args.resume else None
    if resume is not None and resume.get('kind') == 'bc':
        parser.error('An Actor-only BC checkpoint is not a SAC snapshot; use --init-bc instead of --resume')
    bc = load_checkpoint(args.init_bc) if args.init_bc else None
    if bc is not None and bc.get('kind') != 'bc':
        parser.error('--init-bc requires a BC Actor checkpoint')
    value = (resume["config"] if resume else json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config
             else bc['config'] if bc is not None else {})
    config = TrainingConfig.from_dict(value)
    if args.steps is not None:
        config.total_steps = args.steps
    if args.episodes is not None:
        config.max_episodes = args.episodes
    if args.device is not None:
        config.device = args.device
    if args.critic_warmup_updates is not None:
        config.bc_critic_warmup_updates = args.critic_warmup_updates
    config.validate()
    output = args.output or (str(Path(args.resume).resolve().parent) if args.resume
                            else str(RUNS_DIR/'sac_from_bc') if args.init_bc else str(RUNS_DIR/'sac'))
    train(config, output, resume,init_bc=args.init_bc,demonstrations=args.demonstrations)


if __name__ == "__main__":
    main()
