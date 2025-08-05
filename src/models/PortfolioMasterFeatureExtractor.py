import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn
def orthogonal_init(module,gain):
  if isinstance(module, (nn.Linear, nn.Conv2d)):
      nn.init.orthogonal_(module.weight, gain=gain)
      if module.bias is not None:
          module.bias.data.fill_(0.0)

class policy_net(nn.Module):
    def __init__(self,indicator_extractor,stock_num,d_model,output_dim):
        super().__init__()
        self.indicator_extractor = indicator_extractor
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(stock_num * d_model, output_dim)
        self.tanh=nn.Tanh()
        orthogonal_init(self.fc, gain=np.sqrt(2))

    def forward(self,indicator_features):
        x = self.indicator_extractor(indicator_features)
        x = self.fc(self.flatten(x))
        x = self.tanh(x)
        return x


class PortfolioMasterFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

    def forward(self, observations: torch.Tensor):
        assets_features = observations[:,:,:,1:5]                         # batch_size * window_length * stock_num * assets_features
        indicator_features = observations[:, :,:, -8:]                    # batch_size * window_length * stock_num * indicator_features
        return indicator_features,assets_features





