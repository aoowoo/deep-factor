import torch
import torch.nn as nn
from .FeatureMapping import FeatureMapping


class SpatialAttentionLayer(nn.Module):
    def __init__(self, num_nodes, in_features, in_len):
        super(SpatialAttentionLayer, self).__init__()
        self.W1 = nn.Linear(in_len, 1, bias=False)
        self.W2 = nn.Linear(in_features, in_len, bias=False)
        self.W3 = nn.Linear(in_features, 1, bias=False)
        self.V = nn.Linear(num_nodes, num_nodes)

        self.bn_w1 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w3 = nn.BatchNorm1d(num_features=num_nodes)
        self.bn_w2 = nn.BatchNorm1d(num_features=num_nodes)

    def forward(self, inputs):
        # inputs: (batch, num_features, num_nodes, window_len)
        part1 = inputs.permute(0, 2, 1, 3)              # (batch,num_nodes,num_features,window_len)
        part2 = inputs.permute(0, 2, 3, 1)              # (batch,num_nodes,window_len,num_features)
        part1 = self.bn_w1(self.W1(part1).squeeze(-1))  # (batch,num_nodes,num_features)=>batchnorm1d
        part1 = self.bn_w2(self.W2(part1))              # (batch,num_nodes,num_features)=>(batch,num_nodes,window_len)
        part2 = self.bn_w3(self.W3(part2).squeeze(-1)).permute(0, 2, 1)  # (batch,window_len,num_nodes)
        S = torch.softmax(self.V(torch.relu(torch.bmm(part1, part2))), dim=-1)
        return S


class SpatialTemporalAttention(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, kernel_size, layers, dropout=0.1):
        super(SpatialTemporalAttention, self).__init__()
        self.layers = layers
        self.tcns = nn.ModuleList()
        self.sans = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.feature_mapping = FeatureMapping(in_features, hidden_dim ,dropout=dropout)
        self.start_batch_norm2d = nn.BatchNorm2d(num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        receptive_field = 1
        self.supports_len = 0

        additional_scope = kernel_size - 1
        a_s_records = []
        dilation = 1
        for l in range(self.layers):
            tcn_sequence = nn.Sequential(
                nn.Conv2d(in_channels=hidden_dim,
                          out_channels=hidden_dim,
                          kernel_size=(1, kernel_size),
                          dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm2d(hidden_dim)
            )

            self.tcns.append(tcn_sequence)
            self.residual_convs.append(nn.Conv2d(in_channels=hidden_dim,
                                                 out_channels=hidden_dim,
                                                 kernel_size=(1, 1)))
            self.bns.append(nn.BatchNorm2d(hidden_dim))

            dilation *= 2
            a_s_records.append(additional_scope)
            receptive_field += additional_scope
            additional_scope *= 2

        self.receptive_field = receptive_field
        for i in range(layers):
            self.sans.append(SpatialAttentionLayer(num_nodes, hidden_dim, receptive_field - a_s_records[i]))
            receptive_field -= a_s_records[i]

    def forward(self, X):
        X = X.permute(0, 2, 1, 3)
        assert not torch.isnan(X).any(), "feature_mapping前为NaN"
        X = self.feature_mapping(X)  # 映射特征
        x = X.permute(0, 3, 1, 2)    # 调整维度
        assert not torch.isnan(x).any(),"feature_mapping后为NaN"

        x = self.start_batch_norm2d(x)

        for i in range(self.layers):
            residual = self.residual_convs[i](x)
            x = self.tcns[i](x)

            attn_weights = self.sans[i](x)
            x = torch.einsum('bnm, bfml->bfnl', (attn_weights, x))

            x = x + residual[:, :, :, -x.shape[3]:]
            x = self.dropout(x)
            x = self.bns[i](x)

        return x.squeeze(-1).permute(0, 2, 1)

