import numpy as np
import os
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

MODELS={"ppo":PPO}


NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

from DeepIndicator.eval.plotting import plot_training_results
from DeepIndicator.eval.evaluation import evaluate_on_test_env

class PortfolioTrainingCallback(BaseCallback):
    def __init__(self, save_path: str, result_path: str, test_env, verbose: int = 1):
        super(PortfolioTrainingCallback, self).__init__(verbose)
        self.portfolio_values = []  # 存储每个 episode 的 portfolio_value
        self.test_portfolio_values = []  # 存储每个 episode 的测试集 portfolio_value
        self.mean_rank_ic_values = []  # 存储每个 episode 的 mean_rank_ic
        self.turnover_rates = []  # 存储每个 episode 的换手率
        self.save_best_train_path = os.path.join(save_path, 'best_train_model.zip')  # 训练集最优模型保存路径
        self.save_best_test_path = os.path.join(save_path, 'best_test_model.zip')  # 测试集最优模型保存路径
        self.save_last_path = os.path.join(save_path, 'last_model.zip')  # 最后一个模型保存路径
        self.result_path = os.path.join(result_path, 'portfolio_over_episodes.png')
        self.test_result_path = os.path.join(result_path, 'test_portfolio_over_episodes.png')  # 测试集结果保存路径
        self.mean_rank_ic_path = os.path.join(result_path, 'mean_rank_ic_over_episodes.png')
        self.turnover_rate_path = os.path.join(result_path, 'turnover_rate_over_episodes.png')
        self.best_train_portfolio_value = -np.inf
        self.best_test_portfolio_value = -np.inf
        self.episode = 0
        self.test_env = test_env  # 测试环境

        # 用于记录当前 episode 中各环境达到 terminal 时的信息
        self.episode_finished_info = {}  # key: 环境索引，value: 对应 info
        self.num_envs = None  # 后续初始化，假定各步 infos 数量一致
        self.all_started = False  # 所有都重新遍历过一遍之后，赋值为True，可以保存模型了

    def _on_step(self) -> bool:
        self._on_episode_end()
        return True

    def _on_episode_end(self) -> None:
        """
        对每个 step 的 infos 进行检查：
        - 如果 info 中显示某环境达到 terminal，则检查该环境是否已经记录过 terminal。
          如果已经记录，则说明该环境在其它环境还未 terminal 前重复触发，报错。
        - 当所有环境均达到 terminal 时，统计输出本 episode 各环境的结果，
          绘制图像、保存模型，并重置记录供下一 episode 使用。
        """
        infos = self.locals.get('infos')
        if infos is None:
            return

        # 初始化环境数量（假设各步 infos 数量保持一致）
        if self.num_envs is None:
            self.num_envs = len(infos)

        # 检查每个环境的终止状态
        for idx, info in enumerate(infos):
            if info.get('episode_terminal', False):
                # 如果该环境已经记录过 terminal，则报错
                if idx in self.episode_finished_info:
                    raise ValueError(f"Environment {idx} finished terminal twice before all others finished!")
                # 记录该环境终止时的信息
                self.episode_finished_info[idx] = info

        # 只有当所有环境均达到 terminal 后，才视为当前 episode 结束
        if len(self.episode_finished_info) == self.num_envs:
            self.episode += 1
            print("=================================")
            print(f"episode: {self.episode}")
            temp_best_train_portfolio_value = -np.inf

            # 按环境索引顺序输出各环境结果
            for idx in sorted(self.episode_finished_info.keys()):
                info = self.episode_finished_info[idx]
                portfolio_value = info['portfolio_value']
                mean_rank_ic = info['mean_rank_ic']
                env_name = info['environment_identify_name']
                # 假设 transaction_cost_result 在各环境间一致，否则可分别输出
                transaction_cost_result = info.get('transaction_cost_result', None)

                # 计算换手率
                turnover_rate = transaction_cost_result * 1000 if transaction_cost_result is not None else 0.0

                print(f"environment {env_name}/{idx + 1} end_total_asset: {portfolio_value} "
                      f"mean_rank_ic: {mean_rank_ic} transaction_cost_result: {transaction_cost_result} "
                      f"turnover_rate: {turnover_rate}")

                self.portfolio_values.append(portfolio_value)
                self.mean_rank_ic_values.append(mean_rank_ic)
                self.turnover_rates.append(turnover_rate)
                temp_best_train_portfolio_value = max(temp_best_train_portfolio_value, portfolio_value)

            if self.all_started == False:
                self.all_started = True
                return

            # 在测试环境中验证模型
            test_portfolio_value = evaluate_on_test_env(self.model, self.test_env)
            self.test_portfolio_values.append(test_portfolio_value)  # 将测试结果添加到列表中

            # 保存训练集上的最优模型
            if temp_best_train_portfolio_value > self.best_train_portfolio_value:
                self.best_train_portfolio_value = temp_best_train_portfolio_value
                if self.verbose > 0:
                    print(
                        f"Best train result {self.best_train_portfolio_value} hit! Saving new best train model to {self.save_best_train_path}")
                self.model.save(self.save_best_train_path)

            # 保存测试集上的最优模型
            print(f"Test result end_total_asset: {test_portfolio_value}")
            if test_portfolio_value > self.best_test_portfolio_value:
                self.best_test_portfolio_value = test_portfolio_value
                if self.verbose > 0:
                    print(
                        f"Best test result {self.best_test_portfolio_value} hit! Saving new best test model to {self.save_best_test_path}")
                self.model.save(self.save_best_test_path)



            plot_training_results(self.portfolio_values, self.result_path, self.test_portfolio_values, self.test_result_path, self.mean_rank_ic_values, self.mean_rank_ic_path, self.turnover_rates, self.turnover_rate_path)

            if self.episode != 0 and self.episode % 5 == 0:
                self._on_training_end()



    def _on_training_end(self) -> None:
        """
        训练结束后调用该方法保存最后的模型
        """
        print(f"Episode {self.episode} Saving the last model to {self.save_last_path}")
        self.model.save(self.save_last_path)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True

class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
            model_name: object,
            policy: object = "MlpPolicy",
            policy_kwargs: object = None,
            model_kwargs: object = None,
            verbose: object = 1,
            seed: object = None,
            tensorboard_log: object = None,
    ) -> object:
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )

        print("model_kwargs: ",model_kwargs)
        print("policy_kwargs: ",policy_kwargs)

        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs
        )
        return model

    def train_model(self, model, tb_log_name, save_path, result_path, env_test, total_timesteps=5000, verbose=1):
        portfolio_callback = PortfolioTrainingCallback(save_path=save_path, result_path=result_path, test_env=env_test,
                                                       verbose=verbose)
        tb_callback = TensorboardCallback(verbose=verbose)
        callback = CallbackList([portfolio_callback, tb_callback])

        model = model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )

        return model




