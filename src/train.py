if __name__ == '__main__':
    import sys
    import os
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取当前脚本目录的父目录
    parent_dir = os.path.dirname(current_dir)
    print("当前脚本的目录:", current_dir)
    print("当前脚本目录的父目录:", parent_dir)
    sys.path.append(parent_dir)
    print(sys.path)

    from DeepIndicator.tools.agent import DRLAgent as agent
    from DeepIndicator.tools.mfm_env import StockPortfolioEnv
    from .models.PortfolioMasterFeatureExtractor import PortfolioMasterFeatureExtractor
    from .models.PortfolioMasterActorCritic import PortfolioMasterActorCritic
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, _patch_env
    from stable_baselines3.common.monitor import Monitor
    from typing import Any, Callable, Dict, Optional, Type, Union
    import torch
    import gymnasium as gym
    import pandas as pd
    from DeepIndicator.scripts.args import get_args

    args = get_args()

    df = pd.read_csv(os.path.join(parent_dir, args.data_path), parse_dates=['date'])
    df.index = df['date'].factorize()[0]
    train_threshold_time_start = pd.to_datetime(args.train_start_date)
    train_threshold_time_end = pd.to_datetime(args.train_end_date)
    train = df[(df['date'] >= train_threshold_time_start) & (df['date'] <= train_threshold_time_end)]

    multi_indicator_list = ['momentum_3h', 'momentum_24h', 'volume_indicator', 'raw_volume_rank_indicator',
                            'vwap_deviation_indicator', 'price_change_rank_indicator', 'volatility_indicator',
                            'close_volume_corr_indicator']
    stock_dimension = len(train.symbol.unique())

    # 状态空间维度计算
    observation_space = 1 + 1 + 1 + 1 + 1 + len(multi_indicator_list)
    print(f"Stock Dimension: {stock_dimension}, State Space: {observation_space}")

    # 设置日志路径
    RESULTS_DIR = args.results_dir
    SAVE_DIR = args.save_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)  # 确保结果目录存在

    # 环境参数
    env_kwargs = {
        "multi_indicator": multi_indicator_list,
        "initial_amount": args.initial_amount,
        "transaction_cost_pct": args.transaction_cost_pct,
        "window_length": args.window_length,
        "observation_space": observation_space,
        "stock_dim": stock_dimension,
        "action_space": stock_dimension,
        "reward_scaling": args.reward_scaling,
        "short_stoploss_percent": args.short_stoploss_percent,
        "result_path": RESULTS_DIR,
        "make_short": args.make_short,
    }


    def make_vec_env(
            env_id: Union[str, Callable[..., gym.Env]],
            n_envs: int = 1,
            seed: Optional[int] = None,
            start_index: int = 0,
            monitor_dir: Optional[str] = None,
            wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
            env_kwargs: Optional[Dict[str, Any]] = None,
            vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
            vec_env_kwargs: Optional[Dict[str, Any]] = None,
            monitor_kwargs: Optional[Dict[str, Any]] = None,
            wrapper_kwargs: Optional[Dict[str, Any]] = None,
    ) -> VecEnv:
        env_kwargs = env_kwargs or {}
        vec_env_kwargs = vec_env_kwargs or {}
        monitor_kwargs = monitor_kwargs or {}
        wrapper_kwargs = wrapper_kwargs or {}
        assert vec_env_kwargs is not None  # for mypy

        def make_env(rank: int) -> Callable[[], gym.Env]:
            def _init() -> gym.Env:
                # For type checker:
                assert monitor_kwargs is not None
                assert wrapper_kwargs is not None
                assert env_kwargs is not None

                if isinstance(env_id, str):
                    # if the render mode was not specified, we set it to `rgb_array` as default.
                    kwargs = {"render_mode": "rgb_array"}
                    kwargs.update(env_kwargs)
                    try:
                        env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                    except TypeError:
                        env = gym.make(env_id, **env_kwargs)
                else:
                    env = env_id(env_id=rank+1,**env_kwargs)
                    # Patch to support gym 0.21/0.26 and gymnasium
                    env = _patch_env(env)

                if seed is not None:
                    # Note: here we only seed the action space
                    # We will seed the env at the next reset
                    env.action_space.seed(seed + rank)  
                # Wrap the env in a Monitor wrapper
                # to have additional training information
                monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
                # Create the monitor folder if needed
                if monitor_path is not None and monitor_dir is not None:
                    os.makedirs(monitor_dir, exist_ok=True)
                env = Monitor(env, filename=monitor_path, **monitor_kwargs)
                # Optionally, wrap the environment with the provided wrapper
                if wrapper_class is not None:
                    env = wrapper_class(env, **wrapper_kwargs)
                return env

            return _init

        # No custom VecEnv is passed
        if vec_env_cls is None:
            # Default: use a DummyVecEnv
            vec_env_cls = DummyVecEnv

        vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
        # Prepare the seeds for the first reset
        vec_env.seed(seed)
        return vec_env

    def make_env():
        def _init(env_id):
            env=StockPortfolioEnv(df=train,env_id=env_id,**env_kwargs)
            return env
        return _init

    # 初始化多线程训练环境
    env_train=make_vec_env(make_env(),n_envs=args.n_envs,vec_env_cls=SubprocVecEnv)
    print(type(env_train))


    # 初始化代理和模型
    agent = agent(env=env_train)
    if_using_ppo = True

    PPO_PARAMS = {
        "n_steps": args.n_steps,            
        "learning_rate": args.learning_rate,   
        "batch_size": args.batch_size,        
        "clip_range": args.clip_range,         
        "gamma": args.gamma,             
        "gae_lambda": args.gae_lambda,         
        "n_epochs": args.n_epochs,
        "exploration_intensity": args.exploration_intensity,
    }

    optimizer_class=torch.optim.AdamW
    optimizer_kwargs={
        "weight_decay": args.weight_decay,      
        "eps": args.eps              
    }
    
    model_kwargs = {
        "last_activation": args.last_activation,
        "latent_dim_pi": args.latent_dim_pi,
        "latent_dim_vf": args.latent_dim_vf,
        "policy_lstm_hidden_size": args.policy_lstm_hidden_size,
        "policy_lstm_num_layers": args.policy_lstm_num_layers,
        "policy_dropout": args.policy_dropout,
        "critic_embedding_dim": args.critic_embedding_dim,
        "critic_indicator_lstm_hidden_size": args.critic_indicator_lstm_hidden_size,
        "critic_indicator_lstm_num_layers": args.critic_indicator_lstm_num_layers,
        "critic_assets_spatio_hidden_dim": args.critic_assets_spatio_hidden_dim,
        "critic_assets_spatio_kernel_size": args.critic_assets_spatio_kernel_size,
        "critic_assets_spatio_layers": args.critic_assets_spatio_layers,
        "critic_assets_transformer_n_heads": args.critic_assets_transformer_n_heads,
        "critic_assets_transformer_d_k": args.critic_assets_transformer_d_k,
        "critic_assets_transformer_d_v": args.critic_assets_transformer_d_v,
        "critic_assets_transformer_d_ff": args.critic_assets_transformer_d_ff,
        "critic_assets_transformer_d_layers": args.critic_assets_transformer_d_layers,
        "critic_dropout": args.critic_dropout,
    }

    policy_kwargs = {
        "stock_num":10,
        "features_extractor_class": PortfolioMasterFeatureExtractor,
        "ortho_init":True,
        "optimizer_kwargs": optimizer_kwargs,
        "optimizer_class": optimizer_class,
        "model_kwargs": model_kwargs
    }



    model_ppo = agent.get_model("ppo", policy=PortfolioMasterActorCritic, model_kwargs=PPO_PARAMS, policy_kwargs=policy_kwargs)



    if if_using_ppo:
        # 设置日志记录器
        tmp_path = os.path.join(RESULTS_DIR, 'tensorboard')
        new_logger_ppo = configure(tmp_path, ["stdout", "tensorboard"])
        model_ppo.set_logger(new_logger_ppo)

    # 加载测试数据
    test_threshold_time_start = pd.to_datetime(args.test_start_date)
    test_threshold_time_end = pd.to_datetime(args.test_end_date)
    test = df[(df['date'] >= test_threshold_time_start) & (df['date'] <= test_threshold_time_end)]

    # 环境参数
    test_env_kwargs = {
        "multi_indicator": multi_indicator_list,
        "initial_amount": args.initial_amount,
        "transaction_cost_pct": args.transaction_cost_pct,
        "window_length": args.window_length,
        "observation_space": observation_space,
        "stock_dim": stock_dimension,
        "action_space": stock_dimension,
        "reward_scaling": args.reward_scaling,
        "short_stoploss_percent": args.short_stoploss_percent,
        "result_path": RESULTS_DIR,
        "make_short": args.make_short
    }

    # 初始化测试环境
    e_test_gym = StockPortfolioEnv(df=test, **test_env_kwargs, env_id='test', train_begin=False, test_begin=True)
    env_test, _ = e_test_gym.get_sb_env()


    # 训练模型
    trained_ppo = agent.train_model(model=model_ppo,
                                    env_test=env_test,
                                    tb_log_name='ppo',
                                    save_path=SAVE_DIR,
                                    result_path=RESULTS_DIR,
                                    total_timesteps=args.total_timesteps)
