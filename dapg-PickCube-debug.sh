METHOD="dapg"
ENV="PickCube"
GPU="0"

YOUR_LOGGING_DIRECTORY="logs/$METHOD-$ENV-debug"

python3.8 maniskill2_learn/apis/run_rl.py configs/mfrl/$METHOD/maniskill2_pn.py \
--work-dir $YOUR_LOGGING_DIRECTORY --gpu-ids $GPU \
--cfg-options "env_cfg.env_name=$ENV-v0" "env_cfg.obs_mode=pointcloud" \
"env_cfg.n_points=1200" "env_cfg.obs_frame=ee" "env_cfg.control_mode=pd_ee_delta_pose" \
"env_cfg.reward_mode=dense" "rollout_cfg.num_procs=5" \
"agent_cfg.batch_size=64" \
"agent_cfg.demo_replay_cfg.buffer_filenames=/isaac/ManiSkill2-data/rigid_body_envs/$ENV-v0/trajectory.h5" \
"eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
"eval_cfg.num_procs=5" \
"train_cfg.n_steps=200" \
"replay_cfg.sampling_cfg.type=TStepTransition" \
"replay_cfg.sampling_cfg.with_replacement=False" \
"replay_cfg.sampling_cfg.horizon=5" \
"agent_cfg.demo_replay_cfg.sampling_cfg.type=TStepTransition" \
"agent_cfg.demo_replay_cfg.sampling_cfg.with_replacement=False" \
"agent_cfg.demo_replay_cfg.sampling_cfg.horizon=5" 
