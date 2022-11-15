data_file = "/isaac/ManiSkill2-data/rigid_body_envs/PegInsertionSide-v0/trajectory.h5"

import h5py
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

h5 = h5py.File(data_file ,'r')

for k in h5.keys():
    actions = np.array(h5[f"{k}/dict_str_actions"])
    channel_num = actions.shape[1]

    for i in range(channel_num):
        data = actions[:, i:i+1]
        df = pd.DataFrame(data, columns=["column"])
        sns_plot = sns.displot(df, x="column", stat="density")
        # figure = sns_plot.get_figure()
        # figure.savefig(f"{k}_c{i}.png")
        plt.savefig(f"channel_{i}_{k}.png")
     
h5.close()