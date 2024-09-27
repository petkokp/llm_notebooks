import torch.nn as nn
from .layer_norm import LayerNorm
from .causal_self_attention import CausalSelfAttention
from .mlp import MLP

class Block(nn.Module):
    def __init__(self, n_embeddings, bias, n_heads, dropout, context_length):
        super().__init__()
        self.layer_norm_attention = LayerNorm(n_embeddings, bias)
        self.attention = CausalSelfAttention(n_embeddings, n_heads, bias, dropout, context_length)
        self.layer_norm_mlp = LayerNorm(n_embeddings, bias)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attention(self.layer_norm_attention(x))
        x = x + self.mlp(self.layer_norm_mlp(x))
        return x