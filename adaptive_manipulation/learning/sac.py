"""Soft Actor-Critic implemented in PyTorch (no Stable-Baselines dependency)."""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from adaptive_manipulation.core.config import SACConfig


def mlp(input_dim, hidden_sizes, output_dim):
    layers = []
    for width in hidden_sizes:
        layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes):
        super().__init__()
        self.network = mlp(observation_dim, hidden_sizes, 2 * action_dim)

    def distribution_parameters(self, observation):
        mean, log_std = self.network(observation).chunk(2, dim=-1)
        return mean, log_std.clamp(-20, 2)

    def sample(self, observation, deterministic=False):
        mean, log_std = self.distribution_parameters(observation)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        # Stable log(1 - tanh(raw)^2), including saturated actions.
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_prob = (distribution.log_prob(raw) - correction).sum(-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes):
        super().__init__()
        self.network = mlp(observation_dim + action_dim, hidden_sizes, 1)

    def forward(self, observation, action):
        return self.network(torch.cat([observation, action], dim=-1))


class SACAgent:
    def __init__(self, observation_dim, action_dim, config=None, device="cpu"):
        self.config = config or SACConfig()
        self.config.validate()
        self.device = torch.device(device)
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.actor = Actor(observation_dim, action_dim, self.config.hidden_sizes).to(self.device)
        self.q1 = Critic(observation_dim, action_dim, self.config.hidden_sizes).to(self.device)
        self.q2 = Critic(observation_dim, action_dim, self.config.hidden_sizes).to(self.device)
        self.target_q1, self.target_q2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for target in (self.target_q1, self.target_q2):
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.config.learning_rate)
        self.log_alpha = torch.tensor(math.log(self.config.initial_alpha), device=self.device,
                                      requires_grad=self.config.automatic_entropy)
        self.alpha_optimizer = (torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)
                                if self.config.automatic_entropy else None)
        self.target_entropy = (-float(action_dim) if self.config.target_entropy is None
                               else self.config.target_entropy)
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        value = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(value, deterministic=deterministic)
        return action[0].cpu().numpy()

    @torch.no_grad()
    def bellman_target(self, batch):
        action, log_prob = self.actor.sample(batch["next_observations"])
        value = torch.minimum(self.target_q1(batch["next_observations"], action),
                              self.target_q2(batch["next_observations"], action)) - self.alpha * log_prob
        # Task deadlines are terminal. Only external truncations bootstrap.
        return batch["rewards"] + self.config.gamma * (1 - batch["terminated"]) * value

    def update(self, batch, update_actor=True):
        target = self.bellman_target(batch)
        q1 = self.q1(batch["observations"], batch["actions"])
        q2 = self.q2(batch["observations"], batch["actions"])
        with torch.no_grad():
            squared_error = ((q1 - target).square() + (q2 - target).square()).flatten()
            terminal = batch["terminated"].flatten().bool()
            diagnostics = {"q1_mean": float(q1.mean()), "q2_mean": float(q2.mean()),
                           "q_target_mean": float(target.mean()), "q_target_std": float(target.std(unbiased=False)),
                           "q_gap_mean": float((q1 - q2).abs().mean()),
                           "td_abs_mean": float(((q1-target).abs() + (q2-target).abs()).mean() / 2)}
            for suffix, mask in (("terminal", terminal), ("nonterminal", ~terminal)):
                count = int(mask.sum())
                total = float(squared_error[mask].sum())
                diagnostics.update({f"{suffix}_samples": count, f"td_squared_sum_{suffix}": total,
                                    f"td_mse_{suffix}": total / max(count, 1)})
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        action, log_prob = self.actor.sample(batch["observations"])
        with torch.no_grad():
            _, log_std = self.actor.distribution_parameters(batch["observations"])
            entropy = float(-log_prob.mean())
            diagnostics.update({"entropy_mean": entropy, "entropy_error": entropy - self.target_entropy})
            for i in range(self.action_dim):
                diagnostics[f"policy_std_{i}"] = float(log_std[:, i].exp().mean())
                diagnostics[f"policy_action_saturation_{i}"] = float((action[:, i].abs() >= .95).float().mean())
        q = torch.minimum(self.q1(batch["observations"], action), self.q2(batch["observations"], action))
        actor_loss = (self.alpha * log_prob - q).mean()
        if update_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        self.q1.requires_grad_(True)
        self.q2.requires_grad_(True)

        alpha_loss = torch.zeros((), device=self.device)
        if update_actor and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        with torch.no_grad():
            for online, target_net in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
                for parameter, target_parameter in zip(online.parameters(), target_net.parameters()):
                    target_parameter.lerp_(parameter, self.config.tau)
        self.updates += 1
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                "alpha_loss": float(alpha_loss.detach()), "alpha": float(self.alpha),
                "actor_updated": float(update_actor), **diagnostics}

    def state_dict(self):
        return {"observation_dim": self.observation_dim, "action_dim": self.action_dim,
                "config": vars(self.config).copy(), "updates": self.updates,
                **{name: getattr(self, name).state_dict() for name in
                   ("actor", "q1", "q2", "target_q1", "target_q2", "actor_optimizer", "critic_optimizer")},
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_optimizer": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None}

    def load_state_dict(self, state):
        if (state["observation_dim"], state["action_dim"]) != (self.observation_dim, self.action_dim):
            raise ValueError("Checkpoint dimensions differ from agent")
        for name in ("actor", "q1", "q2", "target_q1", "target_q2", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(state[name])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.updates = state["updates"]
