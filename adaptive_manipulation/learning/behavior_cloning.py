"""Actor-only behavior cloning, with a deterministic split by demonstration episode."""

import math

import numpy as np
import torch
from adaptive_manipulation.learning.sac import Actor


class BCPolicy:
    """Same Actor and unnormalized observations as SAC; no Critics or alpha."""

    def __init__(self, observation_dim, action_dim=4, hidden_sizes=(256,256), device='cpu'):
        self.device = torch.device(device)
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.actor = Actor(observation_dim, action_dim, hidden_sizes).to(self.device)

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        tensor = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(tensor, deterministic=deterministic)
        return action[0].cpu().numpy()

    @classmethod
    def from_checkpoint(cls, payload, device='cpu'):
        if payload.get('kind') != 'bc':
            raise ValueError('Expected an Actor-only BC checkpoint')
        policy = cls(payload['observation_dim'],payload['action_dim'],payload['hidden_sizes'],device)
        policy.actor.load_state_dict(payload['actor'])
        policy.actor.eval()
        return policy


def initialize_std(actor, std):
    # MSE optimizes the mean head only. Zero output weights keep std constant
    # even while shared hidden layers change; its rows receive zero MSE gradient.
    head = actor.network[-1]
    action_dim = head.out_features // 2
    with torch.no_grad():
        head.weight[action_dim:].zero_()
        head.bias[action_dim:].fill_(math.log(std))


@torch.no_grad()
def imitation_metrics(actor, observations, actions, batch_size=1024):
    squared, absolute = torch.zeros(actions.shape[1],device=actions.device), torch.zeros(actions.shape[1],device=actions.device)
    for start in range(0,len(actions),batch_size):
        mean, _ = actor.distribution_parameters(observations[start:start+batch_size])
        error = torch.tanh(mean)-actions[start:start+batch_size]
        squared += error.square().sum(0)
        absolute += error.abs().sum(0)
    mse = (squared/len(actions)).cpu().tolist()
    mae = (absolute/len(actions)).cpu().tolist()
    return dict(mse=sum(mse)/len(mse),mae=sum(mae)/len(mae),
                **{f'mse_{i}':value for i,value in enumerate(mse)},
                **{f'mae_{i}':value for i,value in enumerate(mae)})




