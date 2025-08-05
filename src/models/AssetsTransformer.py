import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt
from torch.nn.utils.parametrizations import weight_norm

from .SpatialTemporalAttention import SpatialTemporalAttention


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V,d_k):                                    # Q: [batch_size, n_heads, len_q, d_k]
                                                                       # K: [batch_size, n_heads, len_k, d_k]
                                                                       # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        return context, attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff):
        self.d_model=d_model
        self.d_ff=d_ff

        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,d_k,d_v):
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_k=d_k
        self.d_v=d_v

        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        context, attn = ScaledDotProductAttention()(Q, K, V,self.d_k)  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model).cuda()

    def forward(self, dec_inputs, enc_outputs):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        dec_outputs = self.dropout(dec_outputs)  # Dropout after self-attention

        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs)
        dec_outputs = self.dropout(dec_outputs)  # Dropout after encoder-decoder attention

        dec_outputs = self.pos_ffn(dec_outputs)
        dec_outputs = self.dropout(dec_outputs)  # Dropout after feedforward network

        return self.layer_norm(dec_outputs)  # Apply layer normalization

class Decoder(nn.Module):
    def __init__(self,d_model,n_heads,d_k,d_v,d_ff,d_layers,dropout):
        super(Decoder,self).__init__()

        # batch_size * stock_num * 64
        self.layers=nn.ModuleList([
            DecoderLayer(d_model,n_heads,d_k,d_v,d_ff,dropout) for _ in range(d_layers)
        ])
    def forward(self,dec_inputs,enc_outputs):
        dec_outputs=dec_inputs
        for layer in self.layers:
            dec_outputs= layer(dec_outputs, enc_outputs)
        return dec_outputs


class AssetsTransformer(nn.Module):
    def __init__(self, indicator_extractor,assets_extractor,d_model, n_heads, d_k, d_v, d_ff, d_layers, dropout=0.1):
        super(AssetsTransformer, self).__init__()
        self.indicator_extractor=indicator_extractor
        self.assets_extractor=assets_extractor
        self.decoder = Decoder(d_model, n_heads, d_k, d_v, d_ff, d_layers,dropout)

    def forward(self, indicators_features, assets_features):
        extracted_indicators_features=self.indicator_extractor(indicators_features)
        extracted_assets_features = self.assets_extractor(assets_features)

        dec_outputs = self.decoder(extracted_indicators_features, extracted_assets_features)
        return dec_outputs



