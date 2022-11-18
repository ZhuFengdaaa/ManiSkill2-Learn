data_file = "/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory.h5"
write_data_file = "/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory_disc.h5"

import h5py
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from maniskill2_learn.utils.file import load_hdf5, dump_hdf5

def check_gripper(gripper_actions):
    assert gripper_actions[0] == 1
    flag = 1
    change_idx = None
    for idx, action in enumerate(gripper_actions):
        if action == flag:
            continue
        else:
            assert flag == 1
            flag = -1
            assert action == flag
            change_idx = idx
    if change_idx is not None:
        return change_idx
    assert False

h5 = h5py.File(data_file ,'r')
h5w = h5py.File(write_data_file ,'w')

data = load_hdf5(h5)

for k in data:
    traj = data[k]
    actions = traj["actions"]
    gripper_actions = actions[:, -1]
    change_idx = check_gripper(gripper_actions)
    # new_gripper_actions = np.concatenate([np.full((change_idx,), 1), np.full((gripper_actions.shape[0]-change_idx,), 0)], axis=0)
    new_gripper_actions = np.zeros((gripper_actions.shape[0],))
    new_gripper_actions[change_idx] = 1
    data[k]["actions"][:,-1] = new_gripper_actions

dump_hdf5(data, h5w)

h5.close()
h5w.close()