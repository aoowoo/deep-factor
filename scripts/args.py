"""
关键超参数说明：
- n_envs: 16个并行环境采样,每个环境运行时都随机初始化到不同时间步,并且对于市场因子数据,添加N(0.0.1)噪声
- n_steps: 16个环境采样,rollout_steps走完128步,所以每次走完实际上得到了128*16个数据,总步数增加2048步
- learning_rate: 学习率1e-4,默认3e-4,改小一点,取消target_kl限制
- batch_size: 每次rollout_steps走完,得到了2048个数据,分两批拿1024数据训练,1024/128=8
- gamma: 当前动作对于后续的影响是什么？ 到了第t步,当前的奖励占到这一步的0.1, 0.93的t次方=0.1,t约等于32
- gae_lambda: 0.95,默认0.96~0.98,改成0.95,方差小一点,误差大一点
- n_epochs: 每次采样,更新8次,默认10次
- weight_decay: 设置强化学习常用参数1e-2,缓解过拟合
- eps: 使用AdamW优化器,并且手动设置eps为1e-5
- exploration_intensity: 默认0.7,控制动作分布的标准差,从而影响策略的探索程度给予一定正则化
"""
import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser(description="DeepIndicator Hyperparameters")
    parser.add_argument('--config', type=str, default='scripts/hyper.json', help='Path to the config file')
    
    args, _ = parser.parse_known_args()
    with open(args.config, 'r') as f:
        config = json.load(f)

    common = config['common']
    parser.add_argument('--data-path', type=str, default=common['data_path'])
    parser.add_argument('--results-dir', type=str, default=common['results_dir'])
    parser.add_argument('--save-dir', type=str, default=common['save_dir'])
    parser.add_argument('--initial-amount', type=int, default=common['initial_amount'])
    parser.add_argument('--transaction-cost-pct', type=float, default=common['transaction_cost_pct'])
    parser.add_argument('--window-length', type=int, default=common['window_length'])
    parser.add_argument('--reward-scaling', type=float, default=common['reward_scaling'])
    parser.add_argument('--short-stoploss-percent', type=float, default=common['short_stoploss_percent'])
    parser.add_argument('--make-short', action='store_true', default=common['make_short'])

    # Training arguments
    train = config['train']
    parser.add_argument('--train-start-date', type=str, default=train['train_start_date'])
    parser.add_argument('--train-end-date', type=str, default=train['train_end_date'])
    parser.add_argument('--n-envs', type=int, default=train['n_envs'])
    parser.add_argument('--total-timesteps', type=int, default=train['total_timesteps'])
    
    # PPO Hyperparameters
    ppo = config['ppo']
    parser.add_argument('--n-steps', type=int, default=ppo['n_steps'])
    parser.add_argument('--learning-rate', type=float, default=ppo['learning_rate'])
    parser.add_argument('--batch-size', type=int, default=ppo['batch_size'])
    parser.add_argument('--clip-range', type=float, default=ppo['clip_range'])
    parser.add_argument('--gamma', type=float, default=ppo['gamma'])
    parser.add_argument('--gae-lambda', type=float, default=ppo['gae_lambda'])
    parser.add_argument('--n-epochs', type=int, default=ppo['n_epochs'])
    parser.add_argument('--exploration-intensity', type=float, default=ppo['exploration_intensity'])

    # Optimizer Hyperparameters
    optimizer = config['optimizer']
    parser.add_argument('--weight-decay', type=float, default=optimizer['weight_decay'])
    parser.add_argument('--eps', type=float, default=optimizer['eps'])

    # Testing arguments
    test = config['test']
    parser.add_argument('--test-start-date', type=str, default=test['test_start_date'])
    parser.add_argument('--test-end-date', type=str, default=test['test_end_date'])
    parser.add_argument('--model-path', type=str, default=test['model_path'])

    # Model arguments
    model = config['model']
    parser.add_argument('--last-activation', type=str, default=model['last_activation'])
    parser.add_argument('--latent-dim-pi', type=int, default=model['latent_dim_pi'])
    parser.add_argument('--latent-dim-vf', type=int, default=model['latent_dim_vf'])
    
    policy_net = model['policy_net']
    parser.add_argument('--policy-lstm-hidden-size', type=int, default=policy_net['lstm_hidden_size'])
    parser.add_argument('--policy-lstm-num-layers', type=int, default=policy_net['lstm_num_layers'])
    parser.add_argument('--policy-dropout', type=float, default=policy_net['dropout'])

    critic_net = model['critic_net']
    parser.add_argument('--critic-embedding-dim', type=int, default=critic_net['embedding_dim'])
    
    indicator_lstm = critic_net['indicator_lstm']
    parser.add_argument('--critic-indicator-lstm-hidden-size', type=int, default=indicator_lstm['hidden_size'])
    parser.add_argument('--critic-indicator-lstm-num-layers', type=int, default=indicator_lstm['num_layers'])

    assets_spatio = critic_net['assets_spatio']
    parser.add_argument('--critic-assets-spatio-hidden-dim', type=int, default=assets_spatio['hidden_dim'])
    parser.add_argument('--critic-assets-spatio-kernel-size', type=int, default=assets_spatio['kernel_size'])
    parser.add_argument('--critic-assets-spatio-layers', type=int, default=assets_spatio['layers'])

    assets_transformer = critic_net['assets_transformer']
    parser.add_argument('--critic-assets-transformer-n-heads', type=int, default=assets_transformer['n_heads'])
    parser.add_argument('--critic-assets-transformer-d-k', type=int, default=assets_transformer['d_k'])
    parser.add_argument('--critic-assets-transformer-d-v', type=int, default=assets_transformer['d_v'])
    parser.add_argument('--critic-assets-transformer-d-ff', type=int, default=assets_transformer['d_ff'])
    parser.add_argument('--critic-assets-transformer-d-layers', type=int, default=assets_transformer['d_layers'])
    parser.add_argument('--critic-dropout', type=float, default=critic_net['dropout'])

    args = parser.parse_args()
    return args
