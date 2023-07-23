import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=False):
        Z = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
        if mask:
          mask = torch.tril(torch.ones(Z.shape)).bool().to(Z.device)
          Z = Z.masked_fill(~mask, float('-inf'))
        Z = F.softmax(Z, dim=-1)
        return torch.matmul(Z, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8, d_model=512, d_data=64):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_data = d_data

        self.Q_linears = nn.Linear(d_model, d_data * num_heads)
        self.K_linears = nn.Linear(d_model, d_data * num_heads)
        self.V_linears = nn.Linear(d_model, d_data * num_heads)
        self.attention = ScaledDotProductAttention()
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=False):
        batch_size = Q.size(0)

        Q = self.Q_linears(Q).view(batch_size, -1, self.num_heads, self.d_data).transpose(1, 2)
        K = self.K_linears(K).view(batch_size, -1, self.num_heads, self.d_data).transpose(1, 2)
        V = self.V_linears(V).view(batch_size, -1, self.num_heads, self.d_data).transpose(1, 2)

        Z = self.attention(Q, K, V, mask)
        Z = Z.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_data)

        return self.output_linear(Z)
