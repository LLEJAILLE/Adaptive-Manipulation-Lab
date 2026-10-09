"""Deterministic rollout evaluation shared by BC and SAC workflows."""

import numpy as np

def evaluate_policy(agent, env, seeds):
    if agent.observation_dim != env.observation_space.shape[0]:
        raise ValueError("Policy observation dimension differs from environment (remaining time/steps required)")
    rows = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        while True:
            if env.viewer_closed:
                return rows
            observation, _, terminated, truncated, info = env.step(agent.act(observation, deterministic=True))
            if terminated or truncated:
                rows.append({key: info[key] for key in
                             ("seed", "steps", "total_reward", "success", "final_distance", "elapsed", "reason")})
                break
    return rows

def summarize(rows):
    return {"mean_reward": float(np.mean([r["total_reward"] for r in rows])),
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_distance": float(np.mean([r["final_distance"] for r in rows]))}
