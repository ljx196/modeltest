import torch
import math
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):

        # 第一步 query和key点积算出权重矩阵scores
        scores = torch.matmul(query, key.transpose(-1, -2)) \
            / math.sqrt(query.size(-1))

        # 第二步 对mask掉的attn进行屏蔽，这里是将其score增加到无限大
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 第三步 对scores权重矩阵进行softmax
        p_attn = F.softmax(scores, dim=-1)

        # 第四步 dropout操作
        if dropout is not None:
            p_attn = dropout(p_attn)

        # 第五步 求出最后加权后的矩阵
        return torch.matmul(p_attn, value), p_attn


