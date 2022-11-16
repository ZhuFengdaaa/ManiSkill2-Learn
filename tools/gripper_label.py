data_file = "/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory.h5"
write_data_file = "/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory_disc.h5"

import h5py
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from maniskill2_learn.utils.file import load_hdf5, dump_hdf5

def check_actions(actions):
    gripper_actions = actions[:-1]
    assert gripper_actions[0] == 1
    flag = 1
    for idx, action in enumerate(gripper_actions):
        if action == flag:
            continue
        else:
            assert flag == 1
            flag = -1
            assert action == flag
    return True

h5 = h5py.File(data_file ,'r')
h5w = h5py.File(write_data_file ,'w')

data = load_hdf5(h5)

for k in data:
    traj = data[k]
    actions = traj["actions"]
    assert check_actions(actions)

import ipdb; ipdb.set_trace()

h5.close()
h5w.close()