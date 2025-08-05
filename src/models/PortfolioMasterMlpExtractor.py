from typing import Optional, Union
import gymnasium as gym
import numpy as np
import torch
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.utils import get_device

from .ALSTM import ALSTM
from .AssetsTransformer import AssetsTransformer
from .SpatialTemporalAttention import SpatialTemporalAttention



class AttentionAggregation(nn.Module):
    """
    [batch, num_items, feature_dim]
    [batch, feature_dim]
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        # 一个简单的两层MLP来计算注意力分数
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_scores = self.attention_net(x)
        a_weights = F.softmax(a_scores, dim=1)
        weighted_features = x * a_weights
        aggregated_vector = torch.sum(weighted_features, dim=1)

        return aggregated_vector

def orthogonal_init(module,gain):
  if isinstance(module, (nn.Linear, nn.Conv2d)):
      nn.init.orthogonal_(module.weight, gain=gain)
      if module.bias is not None:
          module.bias.data.fill_(0.0)

class critic_net(nn.Module):
    def __init__(self,assets_transformer,stock_num,d_model,output_dim,dropout=0.1,last_activation="tanh"):
        super().__init__()
        self.assets_transformer = assets_transformer
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(stock_num * d_model, output_dim)
        orthogonal_init(self.fc, gain=np.sqrt(2))
        # 根据 last_activation 选择激活函数
        if last_activation == "tanh":
            self.last_activation = nn.Tanh()
        elif last_activation == "relu":
            self.last_activation = nn.ReLU()
        else:
            raise ValueError("Invalid last activation")
        self.dropout = nn.Dropout(dropout)

    def forward(self,indicators_features,assets_features):
        x = self.assets_transformer(indicators_features,assets_features)
        x = self.last_activation(self.fc(self.flatten(x)))
        x = self.dropout(x)
        return x

class policy_net(nn.Module):
    def __init__(self,indicator_extractor,d_model,output_dim,dropout=0.1,last_activation="tanh"):
        super().__init__()
        self.indicator_extractor = indicator_extractor
        self.fc = nn.Linear(d_model, output_dim)
        orthogonal_init(self.fc, gain=np.sqrt(2))
        if last_activation == "tanh":
            self.last_activation = nn.Tanh()
        elif last_activation == "relu":
            self.last_activation = nn.ReLU()
        else:
            raise ValueError("Invalid last activation")
        self.dropout = nn.Dropout(dropout)
    def forward(self,indicator_features):
        x = self.indicator_extractor(indicator_features)
        x = self.last_activation(self.fc(x))
        x = self.dropout(x)
        return x

class PortfolioMasterMlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        stock_num:int,
        last_activation:str="tanh",
        latent_dim_pi: int = 128,
        latent_dim_vf: int = 128,
        policy_lstm_hidden_size: int = 64,
        policy_lstm_num_layers: int = 2,
        policy_dropout: float = 0.1,
        critic_embedding_dim: int = 64,
        critic_indicator_lstm_hidden_size: int = 32,
        critic_indicator_lstm_num_layers: int = 1,
        critic_assets_spatio_hidden_dim: int = 64,
        critic_assets_spatio_kernel_size: int = 2,
        critic_assets_spatio_layers: int = 3,
        critic_assets_transformer_n_heads: int = 8,
        critic_assets_transformer_d_k: int = 8,
        critic_assets_transformer_d_v: int = 8,
        critic_assets_transformer_d_ff: int = 128,
        critic_assets_transformer_d_layers: int = 1,
        critic_dropout: float = 0.1,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf
        self.stock_num=stock_num
        self.last_activation = last_activation


        # Policy network
        self.policy_net = policy_net(
            ALSTM(d_feat=12, hidden_size=policy_lstm_hidden_size,output_size=latent_dim_pi, num_layers=policy_lstm_num_layers),
            d_model=latent_dim_pi,
            output_dim=latent_dim_pi,
            dropout=policy_dropout,
            last_activation=self.last_activation
        )

        self.critic_net = critic_net(
            AssetsTransformer(
                indicator_extractor=ALSTM(d_feat=8, hidden_size=critic_indicator_lstm_hidden_size,output_size=critic_embedding_dim, num_layers=critic_indicator_lstm_num_layers),
                assets_extractor=SpatialTemporalAttention(num_nodes=stock_num,in_features=4,hidden_dim=critic_assets_spatio_hidden_dim,kernel_size=critic_assets_spatio_kernel_size,layers=critic_assets_spatio_layers),
                d_model=critic_embedding_dim,
                n_heads=critic_assets_transformer_n_heads,
                d_k=critic_assets_transformer_d_k,
                d_v=critic_assets_transformer_d_v,
                d_ff=critic_assets_transformer_d_ff,
                d_layers=critic_assets_transformer_d_layers
            ),
            stock_num=stock_num,
            d_model=critic_embedding_dim,
            output_dim=latent_dim_vf,
            dropout=critic_dropout,
            last_activation=self.last_activation
        )
    def forward(self, features) -> tuple[th.Tensor, th.Tensor]:
        indicator_features,assets_features=features
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(indicator_features,assets_features), self.forward_critic(indicator_features,assets_features)

    def forward_actor(self, indicator_features: th.Tensor,assets_features: th.Tensor) -> th.Tensor:
        # batch_size , stock_num , window_length , feature_size
        combined_features = th.cat((indicator_features,assets_features),dim=-1)
        return self.policy_net(combined_features)

    def forward_critic(self, indicator_features: th.Tensor,assets_features: th.Tensor) -> th.Tensor:
        # batch_size, stock_num * feature_size
        return self.critic_net(indicator_features,assets_features)
