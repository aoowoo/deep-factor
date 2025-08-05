import os
import pandas as pd
from stable_baselines3 import PPO
from DeepIndicator.tools.mfm_env import StockPortfolioEnv
from universal.algos import BAH
from DeepIndicator.scripts.args import get_args
from DeepIndicator.eval.evaluation import run_test_evaluation
from DeepIndicator.eval.metrics import calculate_test_metrics
from DeepIndicator.eval.plotting import plot_test_results

def main():
    args = get_args()

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, '..', args.data_path), parse_dates=['date'])
    df.index = df['date'].factorize()[0]

    multi_indicator_list = ['momentum_3h', 'momentum_24h', 'volume_indicator', 'raw_volume_rank_indicator',
                            'vwap_deviation_indicator', 'price_change_rank_indicator', 'volatility_indicator',
                            'close_volume_corr_indicator']
    stock_dimension = len(df.symbol.unique())

    # State space dimension
    observation_space = 1 + 1 + 1 + 1 + 1 + len(multi_indicator_list)
    print(f"Stock Dimension: {stock_dimension}, State Space: {observation_space}")

    # Set up paths
    RESULTS_DIR = args.results_dir
    SAVE_DIR = args.save_dir
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Environment arguments
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
        "make_short": args.make_short
    }

    # Load test data
    test_threshold_time_start = pd.to_datetime(args.test_start_date)
    test_threshold_time_end = pd.to_datetime(args.test_end_date)
    test = df[(df['date'] >= test_threshold_time_start) & (df['date'] <= test_threshold_time_end)]

    # Initialize test environment
    e_test_gym = StockPortfolioEnv(df=test, **env_kwargs, env_id='test', train_begin=False, test_begin=True, log_day=True)
    env_test, _ = e_test_gym.get_sb_env()

    # Load trained model
    model_save_path = os.path.join(args.model_path)
    trained_ppo = PPO.load(model_save_path)

    # Run evaluation
    dates, portfolio_values, shift_rewards, infos = run_test_evaluation(trained_ppo, env_test)

    # Process results
    dates = [pd.to_datetime(date) for date in dates]
    alpha_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'shift_reward': shift_rewards
    })
    alpha_df = alpha_df.set_index('date')

    # Calculate BAH benchmark
    S = test.pivot(index='date', columns='symbol', values='close')
    bah_result = BAH().run(S)
    bah_df = pd.DataFrame({'portfolio_value': bah_result.equity}, index=bah_result.equity.index)
    
    start_date = max(alpha_df.index.min(), bah_df.index.min())
    end_date = min(alpha_df.index.max(), bah_df.index.max())

    alpha_df = alpha_df.loc[start_date:end_date]
    bah_df = bah_df.loc[start_date:end_date]
    print(f"数据对齐至共同日期范围: {start_date.date()} to {end_date.date()}")
    
    # Calculate metrics
    alpha_cum_returns, alpha_drawdown, alpha_cumulative_reward_percentage, bah_cum_returns, bah_drawdown = calculate_test_metrics(alpha_df, bah_df)

    # Plot results
    plot_test_results(alpha_cum_returns, alpha_drawdown, alpha_cumulative_reward_percentage, bah_cum_returns, bah_drawdown, RESULTS_DIR)

    portfolio_value = infos[0]['portfolio_value']
    transaction_cost_result = infos[0]['transaction_cost_result']
    print(f"end_total_asset: {portfolio_value} transaction_cost_result: {transaction_cost_result}")

if __name__ == '__main__':
    main()
