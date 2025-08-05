import torch as th
import torch.nn as nn
from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims
from torch.distributions import Normal


class CustomActionNetwork(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.action_dim = action_dim
        self.scoring_layer = nn.Linear(hidden_size, 1)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        """
        :param latent: Tensor of shape (batch_size, action_dim, hidden_size)
        :return: Tensor of shape (batch_size, action_dim)
        """
        scores = self.scoring_layer(latent).squeeze(-1)  # (batch_size, action_dim)
        log_std = th.log(scores.detach().std(dim=-1, keepdim=True) + 1e-8).expand_as(scores)
        return (scores,log_std)


class AdeGaussianDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int, exploration_intensity: float = 0.7):
        """
        :param action_dim: 动作维度
        :param exploration_intensity: 用于控制标准差的大小，控制策略的探索程度，给予一定正则化
        """
        super().__init__(action_dim)
        self.exploration_intensity = exploration_intensity
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        """
        :param latent_dim: 输入维度 (hidden_size)
        :return: 自定义的均值网络，不再使用 log_std 作为独立参数 ，暂时给log_std赋值为-1
        """
        action_net = CustomActionNetwork(self.action_dim, latent_dim)
        return action_net, -1

    def  proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor = None):
        """
        :param mean_actions: 由 CustomActionNetwork 计算的动作均值 (batch_size, action_dim)
        :param log_std: 旧参数，不再使用
        :return: self
        """

        std = th.exp(log_std)  # 将对数标准差转换为标准差
        self.distribution = Normal(mean_actions, self.exploration_intensity * std) 
        return self

    def entropy(self) -> th.Tensor:
        """ 返回策略的熵，鼓励探索 """
        return sum_independent_dims(self.distribution.entropy())