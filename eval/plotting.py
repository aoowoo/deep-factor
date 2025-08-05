import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.ticker as mticker

def plot_and_summary_episode(result_path, episode_count, environment_identify_name, date_list, portfolio_list, long_short_list, stock_dim, assets_value_percent_list, stock_names, transaction_cost_list):
    # 假设 date_list 是日期列表，portfolio_list 是收益数据，long_short_list 是信号数据
    date_list = sorted(set(date_list))

    x = mdates.date2num(date_list)
    points = np.array([x, portfolio_list]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = ['green' if signal == 1 else 'red' for signal in long_short_list]
    lc = LineCollection(segments, colors=colors, linewidth=2)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_title('Portfolio Return History')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    plt.savefig(os.path.join(result_path, f'plots/{episode_count}_{environment_identify_name}_portfolio_history.png'))
    plt.close()

    # 绘制各个标的的仓位变化图
    plt.figure(figsize=(12, 6 * stock_dim))  # 设置整个图形的大小，高度根据股票数量调整
    # 从上到下绘制每个股票的仓位变化图以及历史的总持仓标的数量
    for i in range(stock_dim):
        plt.subplot(stock_dim + 1, 1, i + 1)  # 创建子图，每个股票一个子图，从上到下排列
        position = [holding[i] for holding in assets_value_percent_list]  # 提取第i个股票的仓位数据
        plt.plot(date_list, position, label=f'Asset {stock_names[i]}')  # 修正顺序：date_list 作为 x 轴，position 作为 y 轴
        plt.title(f'Position History for Asset {stock_names[i]}')
        plt.xlabel('Date')  # x 轴标签改为 Date
        plt.ylabel('Position Value Percent')
        plt.legend()

    plt.tight_layout()  # 自动调整子图布局，避免标签重叠
    plt.savefig(os.path.join(result_path, f'plots/{episode_count}_{environment_identify_name}_position_value_percents.png'))  # 保存图像
    plt.close()  # 关闭图形

    # 绘制交易成本图
    plt.figure(figsize=(12, 6))
    cumulative_sum = 0  # 初始化累加和为0
    cumulative_transaction_cost_list=[0 for _ in range(len(transaction_cost_list))]
    for i in range(len(transaction_cost_list)):
        cumulative_sum += transaction_cost_list[i]  # 累加当前值
        cumulative_transaction_cost_list[i] = cumulative_sum  # 更新当前值为累加
    plt.subplot(2, 1, 2)
    plt.plot(date_list, cumulative_transaction_cost_list, label='Transaction Costs', color='red')
    plt.title('Transaction Costs')
    plt.xlabel('Date')  # x 轴改为 'Date'
    plt.ylabel('Transaction Cost')
    plt.savefig(os.path.join(result_path, f'plots/{episode_count}_{environment_identify_name}_transaction_costs.png'))
    plt.close()

def plot_training_results(portfolio_values, result_path, test_portfolio_values, test_result_path, mean_rank_ic_values, mean_rank_ic_path, turnover_rates, turnover_rate_path):
    # 绘制并保存训练集 portfolio_value 图像
    plt.figure()
    plt.plot(portfolio_values)
    plt.title('Final Portfolio Value Over Episodes (Train)')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value')
    plt.savefig(result_path)
    plt.close()

    # 绘制并保存测试集 portfolio_value 图像
    plt.figure()
    plt.plot(test_portfolio_values)
    plt.title('Final Portfolio Value Over Episodes (Test)')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value')
    plt.savefig(test_result_path)
    plt.close()

    # 绘制并保存 mean_rank_ic 图像
    plt.figure()
    plt.plot(mean_rank_ic_values)
    plt.title('Mean Rank IC Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Mean Rank IC')
    plt.savefig(mean_rank_ic_path)
    plt.close()

    # 绘制并保存换手率图像
    plt.figure()
    plt.plot(turnover_rates)
    plt.title('Turnover Rate Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Turnover Rate')
    plt.savefig(turnover_rate_path)
    plt.close()

def plot_test_results(alpha_cum_returns, alpha_drawdown, alpha_cumulative_reward_percentage, bah_cum_returns, bah_drawdown, RESULTS_DIR):
    # --- 使用 Matplotlib 绘图 ---
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), sharex=True)

    # 子图 1: 累计回报
    axes[0].plot(alpha_cum_returns.index, alpha_cum_returns, label='DeepIndicator', color='mediumseagreen')
    axes[0].plot(bah_cum_returns.index, bah_cum_returns, label='Benchmark (Buy & Hold)', color='grey')
    axes[0].set_title("Cumulative Returns Comparison")
    axes[0].set_ylabel("Cumulative Returns")
    axes[0].grid(True)
    axes[0].legend()

    # 子图 2: 回撤
    axes[1].fill_between(alpha_drawdown.index, 0, alpha_drawdown, color=(239/255, 83/255, 80/255, 0.7), label='DeepIndicator Drawdown')
    axes[1].fill_between(bah_drawdown.index, 0, bah_drawdown, color=(128/255, 128/255, 128/255, 0.5), label='Benchmark Drawdown')
    axes[1].set_title("Drawdown Comparison")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    axes[1].grid(True)
    axes[1].legend()

    # 子图 3: 累计 Alpha
    axes[2].plot(alpha_cumulative_reward_percentage.index, alpha_cumulative_reward_percentage, label='Cumulative Alpha Return (%)', color='mediumseagreen')
    axes[2].set_title("Cumulative Alpha (%)")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Cumulative Alpha (%)")
    axes[2].grid(True)
    axes[2].legend()

    # --- 保存图表 ---
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    png_path = os.path.join(plots_dir, 'test_strategy_performance.png')
    plt.savefig(png_path, bbox_inches='tight', dpi=600)
    print(f"图表已保存到: {png_path}")
    plt.show()
