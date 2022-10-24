# PYTHONPATH=/isaac/ManiSkill2-Learn:$PYTHONPATH
ENV="StackCube"

python3.8 ./tools/convert_state.py --num-procs 10 --env-name $ENV-v0 --traj-name /isaac/ManiSkill2-data/rigid_body_envs/$ENV-v0/trajectory.h5 --json-name /isaac/ManiSkill2-data/rigid_body_envs/$ENV-v0/trajectory.json --output-name trajectory_pcd.h5
