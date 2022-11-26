import h5py
import json
import math
import torch
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import mani_skill2.envs, gym
import pytorch3d
import pytorch3d.transforms as p3dt
from sapien.core import Pose
from maniskill2_learn.utils.data import GDict

from maniskill2_learn.utils.file import load_hdf5, dump_hdf5
from maniskill2_learn.env import make_gym_env

data_file = "/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory.h5.pcd"

h5 = h5py.File(data_file ,'r')
data = load_hdf5(h5)

with open("/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory.json", "r") as f:
    json_file = json.load(f)

def auto_fix_wrong_name(traj):
    if isinstance(traj, GDict):
        traj = traj.memory
    for key in traj:
        if key in ["action", "reward", "done", "env_level", "next_env_level", "next_env_state", "env_state"]:
            traj[key + "s"] = traj[key]
            del traj[key]
    return traj

reset_kwargs = {}
for d in json_file["episodes"]:
    episode_id = d["episode_id"]
    r_kwargs = d["reset_kwargs"]
    reset_kwargs[episode_id] = r_kwargs

input_dict = {
    "env_name": 'PegInsertionSide-v0',
    "unwrapped": True,
    "obs_mode": 'pointcloud',
    "obs_frame": 'ee',
    "reward_mode": "dense",
    "control_mode": "pd_ee_delta_pose",
    "n_points": 1200,
    "n_goal_points": -1,
}
    
env = make_gym_env(**input_dict)
env_gt = make_gym_env(**input_dict)

input_h5 = h5py.File("/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/pd_ee_delta_pose/trajectory.h5", "r")

# meta data, multi-classification
# joint_moves = [-1, 0, 1] # joint_move[i] * unit
joint_moves = [-5, -1, 0, 1, 5] # joint_move[i] * unit
gripper_moves = [-1, 1] # 1 for open and -1 for close
x_dim = 3 # xyz,abc
a_dim = 3 # xyz,abc
all_dim = x_dim + a_dim
x_unit = 0.02
a_unit = 0.002
disc_action_channels = all_dim + 1
pi = math.pi

def make_disc_action(_dim, _move, gripper_gt):
    disc_action = np.zeros(disc_action_channels)
    disc_action[_dim] = joint_moves.index(_move)
    disc_action[-1] = gripper_moves.index(gripper_gt)
    return disc_action

def disc2cont(disc_action):
    cont_action = np.zeros(disc_action_channels)
    for i in range(disc_action_channels):
        if i < x_dim:
            cont_action[i] = joint_moves[disc_action[i].astype('int')] * x_unit
        elif i < all_dim:
            cont_action[i] = joint_moves[disc_action[i].astype('int')] * a_unit
        else:
            cont_action[i] = gripper_moves[disc_action[i].astype('int')]
    return cont_action

def tcp2axis(tcp_pose):
    tcp_axis = p3dt.quaternion_to_axis_angle(torch.tensor(tcp_pose.q)).numpy()
    return (tcp_pose.p, tcp_axis)

def compute_diff(_dim, tcp_axis, tcp_axis_gt):
    if _dim < x_dim:
        return abs(tcp_axis[0][_dim]-tcp_axis_gt[0][_dim])
    else:
        alpha = tcp_axis[1][_dim - x_dim] % 2*pi
        beta = tcp_axis_gt[1][_dim - x_dim] % 2*pi
        return min(
            abs(alpha-beta), 
            abs(alpha+2*pi-beta), 
            abs(alpha-2*pi-beta), 
        )

def select_disc(env, init_state, tcp_pose_gt, gripper_gt):
    tcp_axis_gt = tcp2axis(tcp_pose_gt)
    # xyz
    move_list = []
    print(f"tcp_axis_gt: {tcp_axis_gt}")
    env.set_state(init_state)
    tcp_pose = env.get_obs()["tcp_pose"]
    print(f"tcp_axis: {tcp2axis(tcp_pose)}")
    tcp_axis_best = []
    # optim xyzabc
    for i in range(all_dim):
    # optim only x
    # for i in range(1):
        diff_list = []
        tcp_axis_list = []
        for _move in joint_moves:
            env.set_state(init_state)
            disc_action = make_disc_action(i, _move, gripper_gt)
            _cont_action = disc2cont(disc_action)
            obs, _, _, _ = env.step(_cont_action)
            tcp_pose = obs["tcp_pose"]
            tcp_axis = tcp2axis(tcp_pose)
            diff = compute_diff(i, tcp_axis, tcp_axis_gt)
            diff_list.append(diff)
            tcp_axis_list.append(tcp_axis)
        move_idx = diff_list.index(min(diff_list))
        move_list.append(move_idx)
        tcp_axis_best.append(tcp_axis_list[move_idx][0][i] if i<x_dim else tcp_axis_list[move_idx][1][i-x_dim])
    print(f"tcp_axis_best: {tcp_axis_best}")
    # move_list += [2]*(a_dim+2) # optim only x
    move_list.append(gripper_moves.index(gripper_gt))
    disc_action = np.array(move_list)
    return disc_action

def save_video(image_list, filename):
    video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 20, (image_list[0].shape[1], image_list[0].shape[0]))
    for img in image_list:
        video_writer.write(img)
    video_writer.release()

cnt = 0
for k in data:
    traj = data[k]
    states = traj["env_states"]
    actions = traj["actions"]
    traj_len = len(actions)
    cur_episode_num = eval(k.split('_')[-1])
    env.reset(**reset_kwargs[cur_episode_num])
    env_gt.reset(**reset_kwargs[cur_episode_num])
    img_list, img_gt_list = [], []

    for i in range(traj_len):
        action = actions[i]
        # assert np.allclose(_state, traj["env_states"][i],atol=1e-2)
        env_gt.set_state(traj["env_states"][-1])
        tcp_pose_gt = env_gt.get_obs()["tcp_pose"]
        _state = env.get_state()
        disc_action = select_disc(env, _state, tcp_pose_gt, action[-1])
        env.set_state(_state)
        _ = env.step(disc2cont(disc_action))
        img = env.render("rgb_array")
        img_gt = env_gt.render("rgb_array")
        img_list.append(img)
        img_gt_list.append(img_gt)
    
    save_video(img_list, f"{k}_xyzabc.mp4")
    save_video(img_gt_list, f"{k}_gt.mp4")
    cnt += 1
    if cnt > 3:
        assert False
    