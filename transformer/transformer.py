import torch.nn as nn
import torch.nn.functional as F

from .positional_embedding import PositionalEmbedding
from .feed_forward import FeedForward
from .attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, num_heads=8, d_model=512, d_data=64):
        super(Encoder, self).__init__()
        self.mha = MultiHeadAttention(num_heads, d_model, d_data)
        self.add_norm_mha = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_model * 4)
        self.add_norm_feed_forward = nn.LayerNorm(d_model)

    def forward(self, X):
        X_mha = self.mha(X, X, X)
        X_mha = self.add_norm_mha(X_mha + X)
        Z = self.feed_forward(X_mha)
        Z = self.add_norm_feed_forward(X_mha + Z)
        return Z

class Decoder(nn.Module):
    def __init__(self, num_heads=8, d_model=512, d_data=64):
        super(Decoder, self).__init__()
        self.masked_mha = MultiHeadAttention(num_heads, d_model, d_data)
        self.add_norm_masked_mha = nn.LayerNorm(d_model)
        self.combined_mha = MultiHeadAttention(num_heads, d_model, d_data)
        self.add_norm_combined_mha = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_model * 4)
        self.add_norm_feed_forward = nn.LayerNorm(d_model)

    def forward(self, X, Z_input):
        X_masked_mha = self.masked_mha(X, X, X, True)
        X_masked_mha = self.add_norm_masked_mha(X_masked_mha + X)
        Z_mha = self.combined_mha(X_masked_mha, Z_input, Z_input)
        Z_mha = self.add_norm_combined_mha(Z_mha + X_masked_mha)
        Z = self.feed_forward(Z_mha)
        Z = self.add_norm_feed_forward(Z_mha + Z)
        return Z

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers=6, num_heads=8, d_model=512):
        super(Transformer, self).__init__()
        if d_model % num_heads != 0:
           raise ValueError("d_model is not divisible by num_heads.")
        d_data = d_model // num_heads
        self.d_model = d_model
        self.src_positional_embedding = PositionalEmbedding(src_vocab_size, d_model)
        self.tgt_positional_embedding = PositionalEmbedding(tgt_vocab_size, d_model)
        self.encoders = nn.ModuleList([Encoder(num_heads, d_model, d_data) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([Decoder(num_heads, d_model, d_data) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, X, Y):
        X = self.src_positional_embedding(X) * self.d_model**0.5
        for encoder in self.encoders:
          X = encoder(X)

        Y = self.tgt_positional_embedding(Y) * self.d_model**0.5
        for decoder in self.decoders:
          Y = decoder(Y, X)

        Y = self.linear(Y)
        return F.log_softmax(Y, dim=-1)