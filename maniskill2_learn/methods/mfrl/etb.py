"""
Proximal Policy Optimization Algorithms (PPO):
    https://arxiv.org/pdf/1707.06347.pdf

Related Tricks(May not be useful):
    Mastering Complex Control in MOBA Games with Deep Reinforcement Learning (Dual Clip)
        https://arxiv.org/pdf/1912.09729.pdf
    A Closer Look at Deep Policy Gradients (Value clip, Reward normalizer)
        https://openreview.net/pdf?id=ryxdEkHtPS
    Revisiting Design Choices in Proximal Policy Optimization
        https://arxiv.org/pdf/2009.10897.pdf

Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations (DAPG):
        https://arxiv.org/pdf/1709.10087.pdf
"""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from maniskill2_learn.env import build_replay
from maniskill2_learn.networks import build_actor_critic, build_model
from maniskill2_learn.utils.torch import build_optimizer
from maniskill2_learn.utils.data import DictArray, GDict, to_np, to_torch
from maniskill2_learn.utils.meta import get_logger, get_world_rank, get_world_size
from maniskill2_learn.utils.torch import BaseAgent, RunningMeanStdTorch, RunningSecondMomentumTorch, barrier, get_flat_grads, get_flat_params, set_flat_grads

from ..builder import MFRL


@MFRL.register_module()
class ETB(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        discriminator_cfg,
        env_params,
        batch_size=256,
        discriminator_batch_size=512,
        discriminator_update_freq=0.125,
        discriminator_update_n=5,
        episode_based_discriminator_update=True,
        **kwargs
    ):
        super(ETB, self).__init__()
        pass

    def update_parameters(self, memory, updates, expert_replay):
        pass

    def update_discriminator_helper(self, expert_replay, recent_traj_replay):
        expert_sampled_batch = expert_replay.sample(self.discriminator_batch_size // 2).to_torch(
            dtype="float32", device=self.device, non_blocking=True
        )
        recent_traj_sampled_batch = recent_traj_replay.sample(self.discriminator_batch_size // 2).to_torch(
            dtype="float32", device=self.device, non_blocking=True
        )
        expert_sampled_batch = self.process_obs(expert_sampled_batch)
        recent_traj_sampled_batch = self.process_obs(recent_traj_sampled_batch)

        expert_out = torch.sigmoid(self.discriminator(expert_sampled_batch["obs"], expert_sampled_batch["actions"]))
        recent_traj_out = torch.sigmoid(self.discriminator(recent_traj_sampled_batch["obs"], recent_traj_sampled_batch["actions"]))

        self.discriminator_optim.zero_grad()
        discriminator_loss = self.discriminator_criterion(
            expert_out, torch.zeros((expert_out.shape[0], 1), device=self.device)
        ) + self.discriminator_criterion(recent_traj_out, torch.ones((recent_traj_out.shape[0], 1), device=self.device))
        discriminator_loss = discriminator_loss.mean()
        discriminator_loss.backward()
        self.discriminator_optim.step()

    def update_discriminator(self, expert_replay, recent_traj_replay, n_finished_episodes):
        if self.episode_based_discriminator_update:
            self.discriminator_counter += n_finished_episodes
        else:
            self.discriminator_counter += 1
        if self.discriminator_counter >= math.ceil(1.0 / self.discriminator_update_freq):
            for _ in range(self.discriminator_update_n):
                self.update_discriminator_helper(expert_replay, recent_traj_replay)
            self.discriminator_counter = 0
            return True
        else:
            return False
