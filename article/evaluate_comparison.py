"""Frozen BC/SAC paired evaluation; no learning or checkpoint selection.

Default protocol: 50 fresh seeds, plus six selection seeds reported separately.
Results are immutable: a nonempty output directory is rejected.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import mujoco
from adaptive_manipulation.core.config import TrainingConfig
from adaptive_manipulation.data.checkpoints import load_checkpoint, check_environment_compatibility
from adaptive_manipulation.data.demonstrations import environment_signature
from adaptive_manipulation.learning.behavior_cloning import BCPolicy
from adaptive_manipulation.learning.sac import SACAgent
from adaptive_manipulation.simulation.gym_env import ManipulationEnv

OUT = ROOT / "article/experiments/heldout_50"
CHECKPOINTS = {"bc": "runs/bc_gamepad/best.pt", "sac": "runs/sac_from_bc/best.pt"}
TEST_SEEDS = list(range(9_000_000, 9_000_050))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_unused_seeds():
    """Audit available configs, demonstration metadata and existing episode CSVs."""
    checked = []
    for path in sorted((ROOT / "runs").rglob("config.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        conf = TrainingConfig.from_dict(raw.get("environment", raw))
        assert all(not conf.is_training_seed(s) and not conf.is_validation_seed(s) for s in TEST_SEEDS), path
        checked.append(str(path.relative_to(ROOT)))
    for path in sorted((ROOT / "demonstrations").rglob("session.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert not set(TEST_SEEDS) & set(raw["seeds"]), path
        checked.append(str(path.relative_to(ROOT)))
    for path in sorted((ROOT / "runs").rglob("*.csv")) + [ROOT / "evaluation.csv"]:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if "seed" not in (reader.fieldnames or []):
                continue
            assert all(int(row["seed"]) not in TEST_SEEDS for row in reader), path
        checked.append(str(path.relative_to(ROOT)))
    return {p: digest(ROOT / p) for p in checked}


def evaluate(policy, env, seeds):
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        initial_distance = float(np.linalg.norm(obs[29:32]))
        spawn = env.core.spawn_position
        components = dict(progress_reward=0., proximity_reward=0., time_penalty=0., terminal_reward=0.)
        while True:
            obs, _, terminated, truncated, info = env.step(policy.act(obs, deterministic=True))
            for key in components:
                components[key] += info[key]
            if terminated or truncated:
                row = {k: info[k] for k in ("seed", "steps", "total_reward", "success", "final_distance", "elapsed", "reason")}
                row.update(initial_distance=initial_distance, spawn_x=spawn[0], spawn_y=spawn[1],
                           **components)
                assert abs(sum(components.values()) - row["total_reward"]) < 1e-7
                rows.append(row)
                break
    return rows


def aggregate(rows):
    successes = [r for r in rows if r["success"]]
    n, k = len(rows), len(successes)
    # Wilson interval: binomial descriptive uncertainty for this fixed policy.
    z = 1.959963984540054
    center = (k / n + z*z / (2*n)) / (1 + z*z/n)
    half = z * np.sqrt((k/n)*(1-k/n)/n + z*z/(4*n*n)) / (1+z*z/n)
    return dict(episodes=n, successes=k, success_rate=k/n,
                success_wilson_95=[float(center-half), float(center+half)],
                mean_return=float(np.mean([r["total_reward"] for r in rows])),
                mean_episode_seconds=float(np.mean([r["elapsed"] for r in rows])),
                mean_contact_seconds=float(np.mean([r["elapsed"] for r in successes])) if successes else None,
                mean_steps=float(np.mean([r["steps"] for r in rows])),
                mean_final_distance=float(np.mean([r["final_distance"] for r in rows])),
                failures=[r["seed"] for r in rows if not r["success"]])


def pair(bc, sac):
    result = []
    for b, s in zip(bc, sac):
        assert b["seed"] == s["seed"] and b["spawn_x"] == s["spawn_x"] and b["spawn_y"] == s["spawn_y"]
        row = {"seed": b["seed"], "bc_success": b["success"], "sac_success": s["success"],
               "bc_seconds": b["elapsed"], "sac_seconds": s["elapsed"],
               "bc_steps": b["steps"], "sac_steps": s["steps"],
               "bc_return": b["total_reward"], "sac_return": s["total_reward"],
               "bc_distance": b["final_distance"], "sac_distance": s["final_distance"]}
        row.update(delta_return=s["total_reward"]-b["total_reward"],
                   delta_seconds=s["elapsed"]-b["elapsed"], delta_steps=s["steps"]-b["steps"])
        for key in ("progress_reward", "proximity_reward", "time_penalty", "terminal_reward"):
            row["delta_"+key] = s[key] - b[key]
        result.append(row)
    return result


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError("Evaluation exists; use a new study directory for another experiment")
    audited = check_unused_seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    payloads = {name: load_checkpoint(ROOT / path) for name, path in CHECKPOINTS.items()}
    hashes = {name: digest(ROOT / path) for name, path in CHECKPOINTS.items()}
    configurations = {name: TrainingConfig.from_dict(p["config"]) for name, p in payloads.items()}
    assert configurations["bc"].environment_kwargs() == configurations["sac"].environment_kwargs()
    validation = payloads["bc"]["split"]["validation_seeds"]
    manifest = dict(protocol="paired_frozen_policies_v1", independent_seeds=TEST_SEEDS,
                    selection_seeds=validation, deterministic=True, device="cpu",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoints={name: dict(path=path, sha256=hashes[name],
                        epoch=payloads[name].get("epoch"),
                        training_step=payloads[name].get("training", {}).get("global_step"))
                        for name, path in CHECKPOINTS.items()},
                    prior_seed_audit=audited, python=platform.python_version(),
                    torch=torch.__version__, mujoco=mujoco.__version__,
                    script_sha256=digest(__file__))
    env = ManipulationEnv(**configurations["bc"].environment_kwargs())
    manifest["environment_signature"] = environment_signature(env)
    # Write the fixed protocol before producing test results.
    (OUT / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    policies = {"bc": BCPolicy.from_checkpoint(payloads["bc"], "cpu")}
    sac = payloads["sac"]["agent"]
    policies["sac"] = SACAgent(sac["observation_dim"], sac["action_dim"], configurations["sac"].sac, "cpu")
    policies["sac"].load_state_dict(sac)
    results = {}
    try:
        for name, policy in policies.items():
            check_environment_compatibility(payloads[name], env)
            for split, seeds in [("independent", TEST_SEEDS), ("validation", validation)]:
                rows = evaluate(policy, env, seeds)
                results[name, split] = rows
                write_csv(OUT / f"{name}_{split}.csv", rows)
                print(name, split, json.dumps(aggregate(rows)), flush=True)
    finally:
        env.close()
    summary = {}
    failures = []
    for split in ("independent", "validation"):
        paired = pair(results["bc", split], results["sac", split])
        write_csv(OUT / f"paired_{split}.csv", paired)
        common = [r for r in paired if r["bc_success"] and r["sac_success"]]
        summary[split] = {name: aggregate(results[name, split]) for name in policies}
        summary[split]["paired"] = dict(common_successes=len(common),
            faster_sac_common=sum(r["delta_seconds"] < 0 for r in common),
            mean_common_delta_seconds=float(np.mean([r["delta_seconds"] for r in common])) if common else None,
            **{key:float(np.mean([r[key] for r in paired])) for key in paired[0] if key.startswith("delta_")})
        for name in policies:
            failures.extend(dict(policy=name, split=split, **r) for r in results[name, split] if not r["success"])
    write_csv(OUT / "failures.csv", failures, ["policy", "split", *results["bc", "independent"][0]])
    assert all(digest(ROOT / path) == hashes[name] for name, path in CHECKPOINTS.items())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
