
agent_cfg = dict(
    type="AABC",
    gamma=0.95,
    lmbda=0.95,
    critic_coeff=0.5,
    entropy_coeff=0,
    critic_clip=False,
    obs_norm=False,
    rew_norm=True,
    adv_norm=True,
    recompute_value=True,
    num_epoch=2,
    critic_warmup_epoch=4,
    batch_size=330,
    detach_actor_feature=False,
    max_grad_norm=0.5,
    eps_clip=0.2,
    max_kl=0.2,
    dual_clip=None,
    shared_backbone=True,
    ignore_dones=True,
    dapg_lambda=0.1,
    dapg_damping=0.995,

    # gail
    discriminator_batch_size=512,
    discriminator_update_freq=0.125,
    discriminator_update_n=5,
    episode_based_discriminator_update=True,
    env_reward_proportion=0.3,    
    update_coeff=0.005,
    alpha=0.2,
    target_update_interval=1,
    automatic_alpha_tuning=True,
    gail_detach_actor_feature=True,
    alpha_optim_cfg=dict(type="Adam", lr=3e-4),
    actor_cfg=dict(
        type="ContinuousActor",
        head_cfg=dict(
            type="GaussianHead",
            init_log_std=-1,
            clip_return=False,
            predict_std=False,
        ),
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["512 + agent_shape", 256, 256, "action_shape"],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4, param_cfg={"(.*?)visual_nn(.*?)": None}),
    ),
    critic_cfg=dict(
        type="ContinuousCritic",
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=None,
            mlp_cfg=dict(
                type="LinearMLP", norm_cfg=None, mlp_spec=["512 + agent_shape + action_shape", 256, 256, 1], inactivated_output=True, zero_init_output=True
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ),
    demo_replay_cfg=dict(
        type="ReplayMemory",
        capacity=-1,
        num_samples=-1,
        keys=["obs", "actions", "dones", "episode_dones"],
        buffer_filenames=[
            "PATH_TO_DEMO.h5",
        ],
    ),
    discriminator_cfg=dict(
        type="ContinuousCritic",
        num_heads=1,
        nn_cfg=dict(
            type="Visuomotor",
            visual_nn_cfg=dict(type="PointNet", feat_dim="pcd_all_channel", mlp_spec=[64, 128, 512], feature_transform=[]),
            mlp_cfg=dict(
                type="LinearMLP",
                norm_cfg=None,
                mlp_spec=["512 + agent_shape + action_shape", 256, 256, 1],
                inactivated_output=True,
                zero_init_output=True,
            ),
        ),
        optim_cfg=dict(type="Adam", lr=3e-4),
    ), 
)


train_cfg = dict(
    on_policy=True,
    total_steps=int(25e6),
    warm_steps=0,
    n_steps=int(2e4), ##############
    n_updates=1,
    n_eval=int(5e6),
    n_checkpoint=int(1e6),
    ep_stats_cfg=dict(
        info_keys_mode=dict(
            success=[True, "max", "mean"],
        )
    ),
)

env_cfg = dict(
    type="gym",
    env_name="xxx-v0",
    obs_mode='pointcloud',
    ignore_dones=True,
)

replay_cfg = dict(
    type="ReplayMemory",
    capacity=int(2e4),
    sampling_cfg=dict(type="OneStepTransition", with_replacement=False),
)

expert_replay_cfg = dict(
    type="ReplayMemory",
    capacity=-1, # auto-adjust capacity based on loaded demonstrations
    buffer_filenames=[
        "PATH_TO_DEMO.h5",
    ],
)

recent_traj_replay_cfg = dict(
    type="ReplayMemory",
    capacity=20000,
)

rollout_cfg = dict(
    type="Rollout",
    num_procs=5,
    with_info=True,
    multi_thread=False,
)

eval_cfg = dict(
    type="Evaluation",
    num_procs=1,
    num=10,
    use_hidden_state=False,
    save_traj=False,
    save_video=True,
    log_every_step=False,
    env_cfg=dict(ignore_dones=False),
)
