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

def save_video(image_list, filename):
    video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 20, (image_list[0].shape[1], image_list[0].shape[0]))
    for img in image_list:
        video_writer.write(img)
    video_writer.release()

cnt = 0
for _, k in enumerate(data):
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
        env_gt.set_state(traj["env_states"][i])
        _ = env.step(action)
        _ = env_gt.step(action)
        img = env.render("rgb_array")
        img_gt = env_gt.render("rgb_array")
        img_list.append(img)
        img_gt_list.append(img_gt)
    
    save_video(img_list, f"{k}_accu.mp4")
    save_video(img_gt_list, f"{k}_noaccu.mp4")
    cnt += 1
    if cnt > 10:
        assert False
    