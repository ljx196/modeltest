import torch.nn as nn
from .attention import Attention


class MultiHeadAttention(nn.Module):
    """
    多头注意力
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 第一步 QKV矩阵先过一遍线性层
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 第二步 通过一次attention操作，获得加权矩阵，x，attn是权值矩阵
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 第三步 使用view操作将结果连接起来
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # 第四步 返回最后一个线性层结果
        return self.output_linear(x)
