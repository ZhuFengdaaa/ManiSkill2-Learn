# Assuming 2 gpus each with 12GB memory; 
# if you have a GPU with more memory (e.g. 24GB), you can set --gpu-ids and --sim-gpu-ids to be the same;
# if you only have one GPU with small memory, then you can set a smaller rollout_cfg.num_procs (e.g. =5)
METHOD="dapg"
ENV="PickCube"
YOUR_LOGGING_DIRECTORY="logs/$METHOD-$ENV-tstep1-log"
GPU=6
SIM_GPU=5

python maniskill2_learn/apis/run_rl.py configs/mfrl/$METHOD/maniskill2_pn.py \
            --work-dir $YOUR_LOGGING_DIRECTORY --gpu-ids $GPU --sim-gpu-ids $SIM_GPU \
            --cfg-options "env_cfg.env_name=$ENV-v0" "env_cfg.obs_mode=pointcloud" "env_cfg.n_points=1200" \
            "rollout_cfg.num_procs=32" "env_cfg.reward_mode=dense" \
            "env_cfg.control_mode=pd_ee_delta_pose" "env_cfg.obs_frame=ee" "env_cfg.n_goal_points=50" \
            "agent_cfg.demo_replay_cfg.buffer_filenames=/isaac/ManiSkill2-data/rigid_body_envs/$ENV-v0/trajectory.h5" \
            "eval_cfg.num=100" "eval_cfg.save_traj=False" "eval_cfg.save_video=True" \
            "train_cfg.total_steps=25000000" "train_cfg.n_checkpoint=5000000" \
            "replay_cfg.sampling_cfg.type=TStepTransition" \
            "replay_cfg.sampling_cfg.with_replacement=False" \
            "replay_cfg.sampling_cfg.horizon=1" \
            "agent_cfg.demo_replay_cfg.sampling_cfg.type=TStepTransition" \
            "agent_cfg.demo_replay_cfg.sampling_cfg.with_replacement=False" \
            "agent_cfg.demo_replay_cfg.sampling_cfg.horizon=1" 
