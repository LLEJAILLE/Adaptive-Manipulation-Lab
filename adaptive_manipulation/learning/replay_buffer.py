"""Fixed-size NumPy replay, with separate termination and truncation flags."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, observation_dim, action_dim, seed=0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.position = self.size = 0
        self.rng = np.random.default_rng(seed)
        self.arrays = {
            "observations": np.empty((capacity, observation_dim), np.float32),
            "actions": np.empty((capacity, action_dim), np.float32),
            "rewards": np.empty((capacity, 1), np.float32),
            "next_observations": np.empty((capacity, observation_dim), np.float32),
            "terminated": np.empty((capacity, 1), np.float32),
            "truncated": np.empty((capacity, 1), np.float32),
        }

    def __len__(self):
        return self.size

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        for key, value in zip(self.arrays, (observation, action, reward, next_observation, terminated, truncated)):
            self.arrays[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        if self.size < batch_size:
            raise ValueError("Not enough transitions")
        indices = self.rng.integers(self.size, size=batch_size)
        return {k: torch.as_tensor(v[indices], device=device) for k, v in self.arrays.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "position": self.position, "size": self.size,
                "rng": self.rng.bit_generator.state,
                "arrays": {k: v[:self.size].copy() for k, v in self.arrays.items()}}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from checkpoint")
        self.size, self.position = state["size"], state["position"]
        for key, value in state["arrays"].items():
            self.arrays[key][:self.size] = value
        self.rng.bit_generator.state = state["rng"]
