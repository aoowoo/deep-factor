import math
import torch
import torch.nn as nn


def orthogonal_init(module,gain):
  if isinstance(module, (nn.Linear, nn.Conv2d)):
      nn.init.orthogonal_(module.weight, gain=gain)
      if module.bias is not None:
          module.bias.data.fill_(0.0)

class FeatureMapping(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.1):
        super(FeatureMapping, self).__init__()
        # 定义全连接层，输入维度为 num_features，输出维度为 hidden_dim
        self.fc = nn.Linear(num_features, hidden_dim)
        orthogonal_init(self.fc, gain=1.0)
        # 使用 tanh 作为激活函数
        self.activation = nn.Tanh()
        # 添加 Dropout 层
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # 通过全连接层进行线性变换
        x = self.fc(x)
        # 应用 tanh 激活函数
        x = self.activation(x)
        # 应用 Dropout
        x = self.dropout(x)
        return x