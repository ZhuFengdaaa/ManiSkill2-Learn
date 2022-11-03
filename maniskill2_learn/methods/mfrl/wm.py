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
import pytorch3d

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
class WM(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        discriminator_cfg,
        env_params,
        batch_size=256,
        discriminator_batch_size=512,
        discriminator_update_n=5,
        episode_based_discriminator_update=True,
        **kwargs
    ):
        super(WM, self).__init__()
        self.discriminator_update_n = discriminator_update_n
        self.discriminator_batch_size = discriminator_batch_size
        self.env_params = env_params
        self.actor = None
        self.critic = None
        discriminator_optim_cfg = discriminator_cfg.pop("optim_cfg")
        self.discriminator = build_model(discriminator_cfg)
        self.discriminator_optim = build_optimizer(self.discriminator, discriminator_optim_cfg)
        self.discriminator_criterion = torch.nn.BCEWithLogitsLoss()
        # self.discriminator_criterion = torch.nn.MSELoss()
        

    def update_parameters(self, memory, updates, expert_replay):
        pass

    def update_discriminator_helper(self, expert_replay):
        self.discriminator_optim.zero_grad()
        expert_sampled_batch = expert_replay.sample(self.discriminator_batch_size // 2).to_torch(
            dtype="float32", device=self.device, non_blocking=True
        )
       
        expert_sampled_batch = self.process_obs(expert_sampled_batch)
        
        # pose label
        def convert_p2m(pose):
            aw, ax, ay, az = torch.unbind(pose, -1)
            m = torch.stack((
                aw, -ax, -ay, -az,
                ax, +aw, -az, +ay,
                ay, +az, +aw, -ax,
                az, -ay, +ax, +aw,
            ), dim=1).view(-1, 4, 4)
            return m

        current_tcp = expert_sampled_batch["obs"]["tcp_pose"]
        next_tcp = expert_sampled_batch["next_obs"]["tcp_pose"]
        current_tcp_pose = current_tcp[:, 3:]
        next_tcp_pose = next_tcp[:, 3:]
        m = convert_p2m(current_tcp_pose)
        m_inv = torch.inverse(m)
        delta_pose = torch.bmm(m_inv, next_tcp_pose.unsqueeze(2)).squeeze()
        # _next_tcp_pose = torch.bmm(m, delta_pose.unsqueeze(2)).squeeze()
        
        # for test
        import pytorch3d
        _next_tcp_pose = pytorch3d.transforms.quaternion_multiply(current_tcp_pose, delta_pose)
        next_tcp_axis = pytorch3d.transforms.quaternion_to_axis_angle(next_tcp_pose)
        _next_tcp_axis = pytorch3d.transforms.quaternion_to_axis_angle(_next_tcp_pose)

        

        # xyz label
        # delta_xyz = 
        
        expert_out = self.discriminator(expert_sampled_batch["obs"], expert_sampled_batch["actions"])
        progress_loss = self.discriminator_criterion(expert_out, expert_sampled_batch["progress"])
        progress_acc = torch.sum(torch.abs(torch.sigmoid(expert_out) - expert_sampled_batch["progress"]) < 0.01) / expert_out.shape[0]
        # expert_out = self.discriminator(expert_sampled_batch["obs"], expert_sampled_batch["actions"])
        # progress_loss = self.discriminator_criterion(torch.sigmoid(expert_out), expert_sampled_batch["progress"])
        # progress_acc = torch.sum(torch.abs(torch.sigmoid(expert_out) - expert_sampled_batch["progress"]) < 0.01) / expert_out.shape[0]

        
        # discriminator_loss = self.discriminator_criterion(
        #     expert_out, torch.zeros((expert_out.shape[0], 1), device=self.device)
        # ) + self.discriminator_criterion(recent_traj_out, torch.ones((recent_traj_out.shape[0], 1), device=self.device))
        # disc_acc = (torch.sum(expert_out < 0.5) + torch.sum(recent_traj_out > 0.5))/(expert_out.shape[0]+recent_traj_out.shape[0])
        # discriminator_loss = discriminator_loss.mean()
        progress_loss.backward()
        self.discriminator_optim.step()
        return {"progress/progress_loss": progress_loss.item(), "progress/progress_acc": progress_acc.item()}

    def update_discriminator(self, expert_replay):
        ret = {}
        n_ep = 0
        n_steps = 0
        for _ in range(self.discriminator_update_n):
            training_info = self.update_discriminator_helper(expert_replay)
            n_ep += np.sum(expert_replay["episode_dones"])
            n_steps += len(expert_replay["progress"])
            for k in training_info.keys():
                if k not in ret:
                    ret[k] = []
                ret[k].append(training_info[k])
        for k in ret:
            ret[k] = np.mean(np.array(ret[k]))
        return True, ret, n_ep, n_steps