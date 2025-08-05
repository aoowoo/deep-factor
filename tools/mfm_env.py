import os
import random

from DeepIndicator.eval.metrics import make_summary_sharpe
import gymnasium as gym
from DeepIndicator.eval.plotting import plot_and_summary_episode
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.utils import seeding
from matplotlib.collections import LineCollection
from scipy.stats import spearmanr
from stable_baselines3.common.vec_env import DummyVecEnv


class StateRelayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = []

    def append(self, item):
        if len(self.items) >= self.max_size:
            self.items.pop(0)
        self.items.append(item)

    def clear(self):
        self.items.clear()

    def is_full(self):
        return len(self.items) == self.max_size

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return str(self.items)


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        initial_amount : int
            start money
        window_length : int
            history memory to use
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        observation_space: int
            the dimension of extracted features after extractor class for ActorCriticPolicy
        action_space: int
            equals stock dimension
        multi_indicators: list
            a list of stocks'relative strengthen technical indicators
        result_path : str
            save the path of the plot

    """
    metadata = {'render.modes': ['console']}

    def __init__(self,
                 env_id,
                 df,
                 stock_dim,
                 initial_amount,
                 window_length,
                 transaction_cost_pct,
                 tunbi_offset_percent,
                 reward_scaling,
                 observation_space,
                 action_space,
                 multi_indicator,
                 short_stoploss_percent,
                 result_path,
                 train_begin=True,
                 test_begin=False,
                 log_day=False,
                 make_short=True,
                 ):
        self.env_id=env_id
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.window_length = window_length
        self.transaction_cost_pct = transaction_cost_pct
        self.tunbi_offset_percent = tunbi_offset_percent
        self.reward_scaling = reward_scaling
        self.observation_space = observation_space
        self.action_space = action_space
        self.multi_indicator = multi_indicator
        self.short_stoploss_percent = short_stoploss_percent
        self.result_path = result_path

        self.stock_names = self.df.loc[self.df.index[0], 'symbol'].values.tolist()
        # action_space after tanh and shape is self.stock_dim
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.action_space,))
        # Shape = (32, 10, 13)
        # features before extracted to extractor_class，and then to input into ActorCritic's mlp_extractor
        # close_memory + assets_holding_memory + assets_holding_value_percent_memory + assets_return_memory + assets_ema_action_score_rank_memory + multi_indicator_memory
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.window_length, self.stock_dim, self.observation_space))  # self.observation_space
        # 用于初始化
        self.initial = True
        # 没有结束
        self.terminal = False
        # 总价值
        self.portfolio_value = self.initial_amount
        # 总价值历史
        self.portfolio_list = []
        # 多空历史
        self.long_short_list = []
        # 换手率历史
        self.transaction_cost_list = []
        # 各个资产占整体价值的百分比，里面的值等价于assets_holding_value_percent_memory
        self.assets_value_percent_list = []
        # 时间历史
        self.date_list=[]

        # 昨天和前天各个标的持仓数量
        self.assets_position_memory = StateRelayMemory(2)
        # 昨天和前天各个标的入场价格
        self.assets_entry_price_memory = StateRelayMemory(2)
        # 各个标的的历史评分
        self.assets_action_score_memory = StateRelayMemory(self.window_length)
        # 各个标的的历史ema评分
        self.assets_ema_action_score_memory = StateRelayMemory(self.window_length)

        # 各个需要用到的状态历史
        self.close_memory = StateRelayMemory(self.window_length)  # 过去window_length各个资产的收盘价
        self.assets_holding_memory = StateRelayMemory(
            self.window_length)  # 过去window_length各个资产是否持仓，做多持仓为1/做空持仓为-1，不持仓则为0
        self.assets_holding_value_percent_memory = StateRelayMemory(self.window_length)  # 过去window_length各个资产占整体价值的比值
        self.assets_return_memory = StateRelayMemory(self.window_length)  # 过去window_length各个资产的相比昨日的收益比例
        self.assets_ema_action_score_rank_memory = StateRelayMemory(
            self.window_length)  # 过去window_length各个资产的评分排名，归一化至【-1~1】
        self.multi_indicator_memory = StateRelayMemory(self.window_length)  # 过去window_length各个资产的多因子数据

        # 做多
        self.longOrShort = 1
        # 天数为第一天
        self.day = self.df.index[0]
        # 初始状态为None
        self.state = None
        # episode为0
        self.episode_count = 0

        self._seed()

        self.environment_identify_name='env_'+str(env_id)

        self.train_begin=train_begin

        self.test_begin=test_begin

        self.make_short=make_short

        self.log_day = log_day

    def reset(
            self,
            *,
            seed=None,
            options=None):

        # 先给所有要用到的变量重新初始化
        self.initial = True
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.portfolio_list.clear()
        self.long_short_list.clear()
        self.transaction_cost_list.clear()
        self.assets_value_percent_list.clear()
        self.date_list.clear()

        # 清空辅助用的StateRelayMemory
        self.assets_position_memory.clear()
        self.assets_entry_price_memory.clear()
        self.assets_action_score_memory.clear()
        self.assets_ema_action_score_memory.clear()

        # 清空状态的StateRelayMemory
        self.close_memory.clear()
        self.assets_holding_value_percent_memory.clear()
        self.assets_return_memory.clear()
        self.assets_ema_action_score_rank_memory.clear()
        self.multi_indicator_memory.clear()

        # 重置天数和状态
        self.day = self.df.index[0]

        if self.train_begin==True:
            self.day=random.randint(self.df.index[0],self.df.index[-1]-50)
            self.train_begin=False

        print(f"Environment {self.environment_identify_name} StartDay {self.day} ")

        self.state = None

        # 先积累window_length的记忆，产生第一次state后，交给ppo产生actions，交给step(actions)
        while self.state is None:
            # 当刚刚初始化时，各个资产的评分都为1e-9分
            self._process_new_day(actions=[1e-9 for _ in range(self.stock_dim)])

        return self.state, {}

    def step(self, actions):

        self._process_new_day(actions)

        if self.terminal:
            self.episode_count += 1
            plot_and_summary_episode(self.result_path, self.episode_count, self.environment_identify_name, self.date_list, self.portfolio_list, self.long_short_list, self.stock_dim, self.assets_value_percent_list, self.stock_names, self.transaction_cost_list)
            if self.test_begin:
                make_summary_sharpe(self.portfolio_list)



        # 在st下做了at动作，返回st+1和rt-1，奖励慢了一步
        # 在st-1下做了at动作，返回st和rt-1，动作快了一拍，对于动作应该使用上一个的
        # st下做了动作at，返回st+1和rt，对于rt需要自己计算下一步的值？

        return self.state, self.reward, self.terminal, False, {'portfolio_value': self.portfolio_value,
                                                               'episode_terminal': self.terminal,
                                                               'transaction_cost_result':sum(self.transaction_cost_list),
                                                               'environment_identify_name':self.environment_identify_name,
                                                               'portfolio_shift_reward':self.portfolio_shift_reward,
                                                               'date': self.data['date'].iloc[0],
                                                               'returns': self.tomorrow_assets_return,
                                                               'scores': self.assets_ema_action_score_memory[-1]
                                                               }




    def calc_portfolio_value(self, position_list, entry_price_list, today_price_list, holding_list, replace):
        # 计算之前的仓位，到今天的价值是多少？如果有空头，可能需要止损
        # 如果replace==True，且发生空头止损，需要修改原position_list和entry_price_list
        portfolio_value = 0
        temp_transaction_cost = 0
        for idx in range(self.stock_dim):
            if holding_list[idx] == 0:
                continue
            elif holding_list[idx] == 1:
                portfolio_value += today_price_list[idx] * position_list[idx]
            else:
                stoploss_price = entry_price_list[idx] * (1.0 + self.short_stoploss_percent)
                if today_price_list[idx] >= stoploss_price:  # 亏损10%，把平仓手续费也考虑进去
                    new_money = entry_price_list[idx] * position_list[idx]
                    stoploss_money = new_money * (1.0 - self.short_stoploss_percent)
                    portfolio_value += stoploss_money
                    if replace == True:  # 止损后，用止损价重新开仓
                        position_list[idx] = stoploss_money / stoploss_price
                        entry_price_list[idx] = stoploss_price
                else:
                    portfolio_value += (entry_price_list[idx] - today_price_list[idx] + entry_price_list[idx]) * \
                                       position_list[idx]
        return portfolio_value, temp_transaction_cost

    def calc_rank_ic(self, score_list, return_list):
        """
        按照评分从高到低分为3档，假设总数为n，则数量为n//3 ， n-(n//3)*2 ， n//3
        将每一档视作一个投资组合，计算投资组合因子和投资组合收益率在截面上的Rank IC。
        每一个投资组合中，按照等权来计算投资组合的收益率和因子取值。
        :param score_list: 因子值列表
        :param return_list: 收益率列表
        :return: Rank IC值
        """
        if len(score_list) != len(return_list):
            raise ValueError("score_list 和 return_list 的长度必须相同")

        n = len(score_list)
        if n < 3:
            raise ValueError("数据量太少，无法分组")

        # 将数据按评分从高到低排序
        sorted_indices = np.argsort(score_list)[::-1]
        sorted_scores = np.array(score_list)[sorted_indices]
        sorted_returns = np.array(return_list)[sorted_indices]

        # 分组
        group_size = n // 3
        group_size_b = n - (n // 3) * 2

        # 计算每个档次的投资组合因子值和收益率
        group1_score = np.mean(sorted_scores[:group_size])
        group1_return = np.mean(sorted_returns[:group_size])

        group2_score = np.mean(sorted_scores[group_size:group_size + group_size_b])
        group2_return = np.mean(sorted_returns[group_size:group_size + group_size_b])

        group3_score = np.mean(sorted_scores[-group_size:])
        group3_return = np.mean(sorted_returns[-group_size:])

        # 构造投资组合因子值和收益率列表
        portfolio_scores = [group1_score, group2_score, group3_score]
        portfolio_returns = [group1_return, group2_return, group3_return]

        if len(set(portfolio_scores)) == 1:
            return 0

        # 计算Rank IC
        rank_ic, _ = spearmanr(portfolio_scores, portfolio_returns)

        return rank_ic

    def _process_new_day(self, actions):
        if self.log_day:
            print(f"today: {self.day} end: {self.df.index[-1]}")
        # 3.16 使用0.5起始的std，除以10之后tanh激活，ema score占比0.4
        actions=np.tanh(np.array(actions)/10)


        self.terminal = True if self.day >= self.df.index[-1]-1 else False

        # 先加载一下基本数据
        self.data = self.df.loc[self.day, :]

        if self.make_short==False:
            self.longOrShort=1
        self.long_short_list.append(self.longOrShort)

        # state = close_memory + assets_holding_memory + assets_holding_value_percent_memory + assets_return_memory + assets_ema_action_score_rank_memory + multi_indicator_memory
        # 与环境交互，处理各个数据，先给能处理的今天的数据赋值
        today_assets_position = [0 for _ in range(self.stock_dim)]
        today_assets_entry_price = [0 for _ in range(self.stock_dim)]
        today_assets_action_score = actions

        today_close = self.data['close'].values.tolist()
        today_assets_holding = [0 for _ in range(self.stock_dim)]
        today_assets_holding_value_percent = [0 for _ in range(self.stock_dim)]
        today_assets_return = [0 for _ in range(self.stock_dim)]
        today_assets_action_score_rank = [0 for _ in range(self.stock_dim)]
        today_multi_indicator = list(zip(*[self.data[indicator].values.tolist() for indicator in self.multi_indicator]))

        transaction_cost = 0

        available_money = 0
        if self.initial == True:
            available_money = self.initial_amount
            assert available_money == self.portfolio_value
            self.initial = False

        # 处理一下，昨天的仓位当中，对于空头如果中途止损，减仓然后重新买入
        if len(self.assets_position_memory)>=1:
            yesterday_close = self.close_memory[-1]
            for idx in range(self.stock_dim):
                today_assets_return[idx] = today_close[idx] / yesterday_close[idx] - 1.0
            today_portfolio, temp_transaction_cost = self.calc_portfolio_value(self.assets_position_memory[-1],
                                                                               self.assets_entry_price_memory[-1],
                                                                               today_close,
                                                                               self.assets_holding_memory[-1],
                                                                               replace=True)
            transaction_cost += temp_transaction_cost / 2


        self.close_memory.append(today_close)
        self.assets_return_memory.append(today_assets_return)
        self.multi_indicator_memory.append(today_multi_indicator)
        self.assets_action_score_memory.append(today_assets_action_score)

        self.assets_ema_action_score_memory.append(today_assets_action_score)
        # 根据today_assets_ema_action_score，给所有标的重新排序，前n//3做多，后n//3做空
        # 从高到低的下标
        ranked_indices = np.argsort(today_assets_action_score)[::-1]
        # i为3 idx为9 第3个的值很小
        # 给today_assets_action_score_rank赋值，-1~1之间均分
        for i, idx in enumerate(ranked_indices):
            # 计算标准化的秩分数，第idx个资产，他的排名是i，i越小，rank_score越大
            rank_score = 1 - (i / (self.stock_dim - 1)) * 2
            today_assets_action_score_rank[idx] = rank_score

        self.assets_ema_action_score_rank_memory.append(today_assets_action_score_rank)

        # 将资产分为 A, B, C
        n_assets = self.stock_dim
        n_A = n_assets // 3  # A组的数量
        n_C = n_A  # C组的数量
        n_B = n_assets - n_A - n_C  # B组的数量

        # A、B、C组
        A_assets = ranked_indices[:n_A]
        B_assets = ranked_indices[n_A:n_A + n_B]
        C_assets = ranked_indices[n_A + n_B:]


        need_to_rebalance = False

        if len(self.assets_position_memory) >= 1:
            today_assets_temp_percent_value = [0 for _ in range(self.stock_dim)]
            for idx in range(self.stock_dim):
                today_assets_position[idx] = self.assets_position_memory[- 1][idx]
                today_assets_entry_price[idx] = self.assets_entry_price_memory[-1][idx]
                today_assets_holding[idx] = self.assets_holding_memory[-1][idx]
                if True:
                    if today_assets_holding[idx] == 1 and (self.longOrShort == -1 or idx not in A_assets):
                        new_money = today_assets_position[idx] * today_close[idx]
                        available_money += new_money
                        today_assets_position[idx] = 0
                        today_assets_entry_price[idx] = 0
                        today_assets_holding[idx] = 0

                    elif today_assets_holding[idx] == -1 and (self.longOrShort == 1 or idx not in C_assets):
                        new_money = (today_assets_entry_price[idx] - today_close[idx] + today_assets_entry_price[idx]) * today_assets_position[idx]
                        available_money += new_money
                        today_assets_position[idx] = 0
                        today_assets_entry_price[idx] = 0
                        today_assets_holding[idx] = 0
                if True:
                    if today_assets_holding[idx] == 1:
                        today_assets_temp_percent_value[idx] = today_assets_position[idx] * today_close[idx] / today_portfolio
                    elif today_assets_holding[idx] == -1:
                        today_assets_temp_percent_value[idx] = ((today_assets_entry_price[idx] - today_close[idx] +today_assets_entry_price[idx]) * today_assets_position[idx]) / today_portfolio
                    if today_assets_temp_percent_value[idx] >= 1.0 / (n_assets // 3) + self.tunbi_offset_percent:
                        need_to_rebalance = True

        if need_to_rebalance:

            target_value = today_portfolio / (n_assets // 3)

            # 确定目标组（多头选A组，空头选C组）
            target_group = A_assets if self.longOrShort == 1 else C_assets

            # 确定哪些不需要加仓
            no_need_to_add = [False for _ in range(self.stock_dim)]

            # 计算总超额资金
            excess_total = 0
            for idx in target_group:
                # 计算当前持仓价值
                if today_assets_holding[idx] == 1:  # 多头
                    current_value = today_assets_position[idx] * today_close[idx]
                    if current_value>target_value:
                        no_need_to_add[idx]=True
                elif today_assets_holding[idx] == -1:  # 空头
                    current_value = (2 * today_assets_entry_price[idx] - today_close[idx]) * today_assets_position[idx]
                    if current_value>target_value:
                        no_need_to_add[idx]=True
                else:
                    continue

                # 累加超出目标值的部分
                excess_total += max(current_value - target_value, 0)

            # 扣除交易费用
            transaction_cost = (excess_total+available_money) * self.transaction_cost_pct / 2
            today_portfolio = today_portfolio - transaction_cost
            new_target_value = today_portfolio / (n_assets // 3)


            for idx in target_group:
                if no_need_to_add[idx]==True:
                    if today_assets_holding[idx] == 1:  # 如果已经是多头持仓
                        current_value = today_assets_position[idx] * today_close[idx]
                    elif today_assets_holding[idx] == -1:
                        current_value = (2 * today_assets_entry_price[idx] - today_close[idx]) * today_assets_position[idx]
                    today_assets_position[idx] *= new_target_value / current_value
                if no_need_to_add[idx] == False:
                    if today_assets_holding[idx] == 1:  # 如果已经是多头持仓
                        current_value = today_assets_position[idx] * today_close[idx]
                        # 需要追加的资金
                        additional_value = new_target_value - current_value
                        if additional_value > 0:
                            # 计算需要追加的仓位
                            additional_position = additional_value / today_close[idx]
                            # 更新持仓量
                            new_position = today_assets_position[idx] + additional_position
                            # 更新入场均价
                            today_assets_entry_price[idx] = ((today_assets_entry_price[idx] * today_assets_position[idx]) +(today_close[idx] * additional_position)) / new_position
                            today_assets_position[idx] = new_position

                    elif today_assets_holding[idx] == -1:  # 如果已经是空头持仓
                        current_value = (2 * today_assets_entry_price[idx] - today_close[idx]) * today_assets_position[idx]
                        # 需要追加的资金
                        additional_value = new_target_value - current_value
                        if additional_value > 0:
                            # 计算需要追加的仓位
                            additional_position = additional_value / (2 * today_close[idx] - today_close[idx])
                            # 更新持仓量
                            new_position = today_assets_position[idx] + additional_position
                            # 更新入场均价
                            today_assets_entry_price[idx] = ((today_assets_entry_price[idx] * today_assets_position[idx]) +(today_close[idx] * additional_position)) / new_position
                            today_assets_position[idx] = new_position

                    else:  # 如果是新开仓
                        if self.longOrShort == 1:  # 做多
                            # 计算可以买入的数量
                            position = new_target_value / today_close[idx]
                            today_assets_position[idx] = position
                            today_assets_entry_price[idx] = today_close[idx]
                            today_assets_holding[idx] = 1
                        else:  # 做空
                            # 计算可以做空的数量
                            position = new_target_value / today_close[idx]
                            today_assets_position[idx] = position
                            today_assets_entry_price[idx] = today_close[idx]
                            today_assets_holding[idx] = -1


        else:
            # TODO: 如果不需要再平衡，那么会对所有被清仓的标的，均匀分配到新开仓的标的上，被清仓标的资金加起来为available_money
            num_count = 0
            for idx in range(self.stock_dim):
                if today_assets_holding[idx] != 1 and self.longOrShort == 1 and idx in A_assets:
                    num_count += 1
                if today_assets_holding[idx] != -1 and self.longOrShort == -1 and idx in C_assets:
                    num_count += 1

            if num_count != 0:
                transaction_cost += available_money * self.transaction_cost_pct / 2
                available_money -= transaction_cost
                available_money /= num_count
                for idx in range(self.stock_dim):
                    if today_assets_holding[idx] != 1 and self.longOrShort == 1 and idx in A_assets:
                        today_assets_holding[idx] = 1
                        today_assets_entry_price[idx] = today_close[idx]
                        today_assets_position[idx] = available_money / today_assets_entry_price[idx]
                    if today_assets_holding[idx] != -1 and self.longOrShort == -1 and idx in C_assets:
                        today_assets_holding[idx] = -1
                        today_assets_entry_price[idx] = today_close[idx]
                        today_assets_position[idx] = available_money / today_assets_entry_price[idx]

        today_assets_true_value = [0 for _ in range(self.stock_dim)]
        for idx in range(self.stock_dim):
            if today_assets_holding[idx] == 1:
                today_assets_true_value[idx] = today_assets_position[idx] * today_close[idx]
            if today_assets_holding[idx] == -1:
                today_assets_true_value[idx] = (today_assets_entry_price[idx] - today_close[idx] +
                                                today_assets_entry_price[idx]) * \
                                               today_assets_position[idx]

        today_end_portfolio = sum(today_assets_true_value)
        for idx in range(self.stock_dim):
            today_assets_holding_value_percent[idx] = today_assets_true_value[idx] / today_end_portfolio

        self.portfolio_value = today_end_portfolio
        self.portfolio_list.append(self.portfolio_value)
        self.date_list.append(self.data['date'].iloc[0])
        transaction_cost = transaction_cost / self.portfolio_value
        self.transaction_cost_list.append(transaction_cost)
        self.assets_value_percent_list.append(today_assets_holding_value_percent)

        self.assets_position_memory.append(today_assets_position)
        self.assets_entry_price_memory.append(today_assets_entry_price)
        self.assets_holding_memory.append(today_assets_holding)
        self.assets_holding_value_percent_memory.append(today_assets_holding_value_percent)



        # 奖励：今天做的调仓动作，比起不调仓，在明天收盘时能得到的奖励是多少？
        tomorrow_close = self.df.loc[self.day+1, 'close'].values.tolist()
        self.tomorrow_assets_return=[tomorrow_close[idx]/today_close[idx]-1.0 for idx in range(self.stock_dim)]
        if len(self.assets_position_memory) >= 2:

            # 首先，计算一下今天各个assets_ema_score和明天的真实收益率的RankIC值
            tomorrow_portfolio, _ = self.calc_portfolio_value(self.assets_position_memory[-1],
                                                                               self.assets_entry_price_memory[-1],
                                                                               tomorrow_close,
                                                                               self.assets_holding_memory[-1],
                                                                               replace=False)

            rank_ic_reward = self.calc_rank_ic(self.assets_ema_action_score_memory[-1], self.tomorrow_assets_return)

            # 再处理一下昨天的仓位如果今天不处理，到明天的portfolio，并且计算一下reward =>  st下做at得到st+1,rt
            today_initial_portfolio, _ = self.calc_portfolio_value(self.assets_position_memory[-2],
                                                                                   self.assets_entry_price_memory[-2],
                                                                                   tomorrow_close,
                                                                                   self.assets_holding_memory[-2],
                                                                                   replace=False)

            self.portfolio_shift_reward = (tomorrow_portfolio / today_initial_portfolio ) - 1.0
            portfolio_reward = (tomorrow_portfolio / today_end_portfolio) - 1.0

            mixed_reward = (np.exp(rank_ic_reward) * self.portfolio_shift_reward + portfolio_reward) * self.reward_scaling + rank_ic_reward
            self.reward = mixed_reward


        if self.close_memory.is_full():
            self._build_state()

        if self.terminal==True:
            # 如果是最后一天了，填补一下明天的数据
            self.long_short_list.append(self.longOrShort)
            self.portfolio_list.append(tomorrow_portfolio)
            self.date_list.append(self.df.loc[self.day+1, 'date'].iloc[0])
            self.assets_value_percent_list.append(today_assets_holding_value_percent)
            self.transaction_cost_list.append(transaction_cost)

        self.day += 1

    def _build_state(self):

        # 检查所有状态长度是否等于32
        lengths = [
            len(self.close_memory),                                 # 32*10
            len(self.assets_holding_memory),                        # 32*10
            len(self.assets_holding_value_percent_memory),          # 32*10
            len(self.assets_return_memory),                         # 32*10
            len(self.assets_ema_action_score_rank_memory),          # 32*10
            len(self.multi_indicator_memory)                        # 32*10*8
        ]

        if len(set(lengths)) > 1 or len(self.close_memory) != self.window_length:
            raise ValueError("所有state组成部分的长度必须相同！且需要等于window_length")

        # 32 * 10 * (1+1+1+1+1+8)
        # state = close_memory + assets_holding_memory + assets_holding_value_percent_memory + assets_return_memory + assets_ema_action_score_rank_memory + multi_indicator_memory
        self.state = [[[] for _ in range(self.stock_dim)] for _ in range(self.window_length)]

        for i in range(self.window_length):
            for j in range(self.stock_dim):
                self.state[i][j].append(self.close_memory[i][j])
                self.state[i][j].append(self.assets_holding_memory[i][j])
                self.state[i][j].append(self.assets_holding_value_percent_memory[i][j])
                self.state[i][j].append(self.assets_return_memory[i][j])
                self.state[i][j].append(self.assets_ema_action_score_rank_memory[i][j])
                for x in self.multi_indicator_memory[i][j]:
                    self.state[i][j].append(x)
        # window_length * stock_num * feature_size => window_length * feature_size * stock_num
        self.state = torch.tensor(self.state).permute(0,2,1)
        # window_length * feature_size * stock_num => layernorm一下 再permute => window_length * stock_num * feature_size
        self.state = np.array(F.layer_norm(self.state,normalized_shape=[self.state.shape[-1]]).permute(0,2,1))

        assert len(self.state.shape)==3,"state张量维度不为3!"
        assert not np.isnan(self.state).any(),"state为nan"

        if not self.test_begin:
            # 添加高斯噪声
            noise = np.random.normal(loc=0.0, scale=0.1, size=self.state.shape)  # 生成与state形状相同的噪声矩阵
            self.state = self.state + noise  # 将噪声加到state上



    def render(self):
        if self.render_mode=='console':
            pass


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs