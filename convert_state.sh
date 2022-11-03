# PYTHONPATH=/isaac/ManiSkill2-Learn:$PYTHONPATH
ENV="PegInsertionSide"

python3.8 ./tools/convert_state.py --num-procs 10 --env-name $ENV-v0 --traj-name /isaac/ManiSkill2-data/rigid_body_envs/$ENV-v0/trajectory.h5 \
--json-name /isaac/ManiSkill2-data/rigid_body_envs/$ENV-v0/trajectory.json \
--output-name trajectory_pcd.h5 \
--control-mode=pd_ee_delta_pose \
--max-num-traj=-1 \
--obs-mode=pointcloud \
--reward-mode=dense \
--obs-frame=ee \
--n-points=1200 \
--with-next