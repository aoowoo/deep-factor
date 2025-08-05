import torch
from torch import nn


class ALSTM(nn.Module):
    def __init__(self, d_feat=8, hidden_size=32,output_size=64, num_layers=1, dropout=0.1, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.output_size=output_size
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=self.output_size)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        # 假设 inputs 的形状为 (batch_size, seq_len, num_nodes, d_feat)
        inputs = inputs.transpose(1, 2)         # 交换维度，形状变为 (batch_size, num_nodes, seq_len, d_feat)
        batch_size, num_nodes, seq_len, d_feat = inputs.shape

        # 将 num_nodes 维度与 batch_size 合并，每个节点单独处理
        x = inputs.reshape(-1, seq_len, d_feat)  # 使用 reshape 替代 view，形状变为 (batch_size*num_nodes, seq_len, d_feat)

        # 经过输入全连接层和激活函数
        x = self.net(x)  # 形状变为 (batch_size*num_nodes, seq_len, hidden_size)

        # RNN 层处理
        rnn_out, _ = self.rnn(x)  # 形状为 (batch_size*num_nodes, seq_len, hidden_size)

        # 如果需要注意力加权，可以计算注意力得分
        attention_score = self.att_net(rnn_out)  # 形状 (batch_size*num_nodes, seq_len, 1)
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)  # 得到注意力加权后的特征，形状 (batch_size*num_nodes, hidden_size)

        output = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att), dim=1)
        )
        # 如果希望恢复到 (batch_size, num_nodes, output_size)
        output = output.reshape(batch_size, num_nodes, self.output_size)  # 使用 reshape 替代 view
        return output