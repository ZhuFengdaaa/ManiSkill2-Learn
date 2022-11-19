data_file = "/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory.h5"
write_data_file = "/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory_disc.h5"

import h5py
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import mani_skill2.envs, gym

from maniskill2_learn.utils.file import load_hdf5, dump_hdf5



h5 = h5py.File(data_file ,'r')
h5w = h5py.File(write_data_file ,'w')

data = load_hdf5(h5)

# def cont2disc_action(actions):
    
env = gym.make('PegInsertionSide-v0', obs_mode='pointcloud')

for k in data:
    traj = data[k]
    import ipdb; ipdb.set_trace()
    actions = traj["actions"]
    states = traj["obs"]["state"]
    # state
    traj_len = len(states)
    for idx in range(traj_len):
        env.set_state(states[idx])
    disc_actions = cont2disc_action(actions)

dump_hdf5(data, h5w)

h5.close()
h5w.close()