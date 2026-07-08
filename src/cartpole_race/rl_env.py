"""Gymnasium env wrapping the EXACT n=7 saturated cart-pole for RL.

Dynamics: TorchCartPole (verified clone of CasADi NLinkCartPole, same masses/
lengths/gravity, same 150 N cart-force clip, same 1 ms ZOH + 0.25 ms RK4).
Training-only; the GATE uses CasADi rollout_zoh directly.

Reward: shaped toward all 7 links upright + cart centered + low velocity,
with a hold bonus when inside the locked success set
(|theta|<5deg, |thetad|<0.5, |x|<2, |xdot|<0.5).

Curriculum: start IC noise can be set near-upright and expanded toward
hanging via set_curriculum(level in [0,1]); level 0 = near upright,
level 1 = full hanging start with IC noise.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from cartpole_race.env_spec import CartPoleSpec
from cartpole_race.torch_dynamics import TorchCartPole


class CartPoleSwingUpVecEnv:
    """Hand-rolled vectorized env (B parallel cart-poles) on torch.

    Avoids gymnasium VectorEnv overhead; steps B envs at once on the torch
    dynamics. Exposes a minimal SB3-compatible VecEnv-like interface used by
    our own PPO loop (we run a clean PPO, not SB3's, for full control).
    """

    def __init__(self, n_links: int = 7, num_envs: int = 256,
                 episode_seconds: float = 8.0, device: str = "cpu",
                 seed: int = 0):
        self.spec = CartPoleSpec().with_n_links(n_links)
        self.n = n_links
        self.nx = self.spec.nx
        self.B = num_envs
        self.device = device
        self.dyn = TorchCartPole(self.spec, device=device, dtype=torch.float32)
        self.control_dt = self.spec.control_dt_s
        self.max_steps = int(round(episode_seconds / self.control_dt))
        self.fbound = self.spec.force_bound_n
        self.rng = np.random.default_rng(seed)
        self.g = torch.Generator(device=device).manual_seed(seed)
        # success tolerances (locked predicate)
        self.theta_tol = np.deg2rad(5.0)
        self.thetad_tol = 0.5
        self.x_tol = 2.0
        self.xdot_tol = 0.5
        self.hold_target = int(round(5.0 / self.control_dt))  # 5000 ticks
        self.curriculum = 1.0  # 1 = hanging start
        # link reach (for tip-height reward): cumulative cos heights
        self.ll = torch.tensor(self.spec.link_lengths_m, device=device,
                               dtype=torch.float32)
        # obs = [cart_pos, cart_vel, all link angvels] + sin(theta) + cos(theta)
        #      = (1 + 1 + n) + n + n  = 2 + 3n
        self.obs_dim = 2 + 3 * self.n
        self.act_dim = 1

    def set_curriculum(self, level: float):
        self.curriculum = float(np.clip(level, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _sample_ic(self, B):
        """IC per curriculum. level 0: near upright; level 1: hanging + noise."""
        x = torch.zeros(B, self.nx, device=self.device, dtype=torch.float32)
        lvl = self.curriculum
        # base angle interpolates upright(0) -> hanging(pi)
        base = lvl * np.pi
        # Level 0 starts INSIDE the measured 150 N basin floor (~0.01 deg =
        # 1.7e-4 rad) so the easy stage is physically holdable; noise grows to
        # full hanging by level 1. This gives PPO a learnable foothold and lets
        # us measure how far up the curriculum it can climb before the 150 N
        # authority wall (the n=7 mechanism) stops it.
        ang0 = 1.5e-4  # ~0.0086 deg, inside basin
        ang1 = np.pi
        ang_noise = ang0 + (0.15 - ang0) * lvl
        x[:, 1:1 + self.n] = base + torch.randn(B, self.n, device=self.device,
                                                generator=self.g) * ang_noise
        x[:, 0] = torch.randn(B, device=self.device, generator=self.g) * (0.01 + 0.1 * lvl)
        x[:, 1 + self.n:] = torch.randn(B, self.nx - 1 - self.n,
                                        device=self.device,
                                        generator=self.g) * (1e-4 + 0.12 * lvl)
        return x

    def reset(self):
        self.x = self._sample_ic(self.B)
        self.t = 0
        self.hold = torch.zeros(self.B, device=self.device)
        return self._obs()

    def _obs(self):
        th = self.x[:, 1:1 + self.n]
        rest = torch.cat([self.x[:, :1], self.x[:, 1 + self.n:]], dim=1)
        return torch.cat([rest, torch.sin(th), torch.cos(th)], dim=1)

    def _tip_potential(self):
        """Sum of link-COM heights normalized; +1 all up, -1 all down."""
        th = self.x[:, 1:1 + self.n]
        # cumulative height of each COM ~ uses cos; reward all cos -> 1
        return torch.cos(th).mean(dim=1)

    def _in_success(self):
        th = self.x[:, 1:1 + self.n]
        thd = self.x[:, 2 + self.n:]
        xc = self.x[:, 0]
        xdc = self.x[:, 1 + self.n]
        ang_ok = (th.abs() <= self.theta_tol).all(dim=1)
        # wrap angle: theta near 0 only (upright). |th|<=tol already handles wrap.
        thd_ok = (thd.abs() <= self.thetad_tol).all(dim=1)
        return ang_ok & thd_ok & (xc.abs() <= self.x_tol) & (xdc.abs() <= self.xdot_tol)

    def step(self, action):
        """action: (B,) in [-1,1] -> force in [-fbound,fbound]."""
        u = torch.as_tensor(action, device=self.device,
                            dtype=torch.float32).reshape(-1) * self.fbound
        self.x = self.dyn.step_zoh(self.x, u)
        self.t += 1
        th = self.x[:, 1:1 + self.n]
        thd = self.x[:, 2 + self.n:]
        xc = self.x[:, 0]
        xdc = self.x[:, 1 + self.n]
        # --- reward shaping ---
        up = torch.cos(th).mean(dim=1)                  # [-1,1], 1=upright
        ang_pen = (th ** 2).mean(dim=1)                 # 0 at upright
        thd_pen = (thd ** 2).mean(dim=1)
        x_pen = xc ** 2
        xd_pen = xdc ** 2
        in_succ = self._in_success()
        self.hold = torch.where(in_succ, self.hold + 1.0,
                                torch.zeros_like(self.hold))
        # BOUNDED reward (keeps the value function stable -- unbounded fall
        # penalties were destabilizing PPO). Per-step reward in roughly [0,2]:
        # a sharp upright peak + an alive-in-set bonus. exp(-angle^2) gives a
        # gradient toward the exact catch; in_succ adds the hold incentive.
        sharp_up = torch.exp(-0.5 * (th ** 2).sum(dim=1) / (np.deg2rad(8.0) ** 2))
        reward = (0.5 * (up + 1.0) * 0.5            # [0,0.5], smooth up signal
                  + 1.0 * sharp_up                   # [0,1], sharp catch peak
                  + 0.5 * in_succ.float()            # in-set alive bonus
                  - 0.02 * torch.clamp(thd_pen, 0, 5)
                  - 0.01 * torch.clamp(x_pen, 0, 5))
        # off-track termination (rail at +/-10 m): mild bounded penalty
        off_track = xc.abs() > self.spec.track_half_length_m
        reward = reward - 1.0 * off_track.float()
        done_time = self.t >= self.max_steps
        held = self.hold >= self.hold_target
        reward = reward + 5.0 * held.float()  # bounded terminal success bonus
        dones = off_track | held | done_time
        info = {"in_success": in_succ, "hold": self.hold.clone(),
                "held": held, "off_track": off_track}
        # auto-reset done envs
        if dones.any():
            idx = dones.nonzero(as_tuple=True)[0]
            newic = self._sample_ic(len(idx))
            self.x[idx] = newic
            self.hold[idx] = 0.0
        if done_time:
            self.t = 0
        return self._obs(), reward, dones, info
