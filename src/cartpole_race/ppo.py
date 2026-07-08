"""Clean continuous-action PPO for the n-link cart-pole swing-up.

Gaussian policy, GAE-lambda, clipped surrogate, value-function clipping,
entropy bonus. Runs on the hand-rolled vectorized torch env (rl_env). No
SB3 dependency at train time (kept minimal and fully under our control), but
torch/gymnasium/sb3 are installed and available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, init_log_std=-0.5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim) + init_log_std)
        # small initial policy output so a fresh net is near-zero-force
        nn.init.zeros_(self.pi_head.bias)
        with torch.no_grad():
            self.pi_head.weight.mul_(0.01)

    def forward(self, obs):
        h = self.shared(obs)
        return self.pi_head(h), self.v_head(h).squeeze(-1)

    def act_mean(self, obs):
        h = self.shared(obs)
        return torch.tanh(self.pi_head(h))

    def get_action(self, obs):
        mean, v = self.forward(obs)
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        dist = Normal(mean, std)
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(-1)
        act = torch.tanh(raw)
        # tanh correction
        logp = logp - torch.log(1 - act.pow(2) + 1e-6).sum(-1)
        return act, logp, v

    def evaluate(self, obs, act):
        mean, v = self.forward(obs)
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        dist = Normal(mean, std)
        # invert tanh
        raw = torch.atanh(act.clamp(-0.999999, 0.999999))
        logp = dist.log_prob(raw).sum(-1)
        logp = logp - torch.log(1 - act.pow(2) + 1e-6).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, v, ent


def compute_gae(rewards, values, dones, last_val, gamma=0.99, lam=0.95):
    """rewards/values/dones: (T,B). last_val:(B,). Returns adv,ret (T,B)."""
    T, B = rewards.shape
    adv = torch.zeros(T, B, device=rewards.device)
    lastgae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        nextval = last_val if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * nextval * nonterm - values[t]
        lastgae = delta + gamma * lam * nonterm * lastgae
        adv[t] = lastgae
    ret = adv + values
    return adv, ret
