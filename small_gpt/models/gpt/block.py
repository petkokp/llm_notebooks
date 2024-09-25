import torch.nn as nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, n_embeddings, n_heads, context_length, dropout):
        super().__init__()
        head_size = n_embeddings // n_heads
        self.self_attention = MultiHeadAttention(n_heads, head_size, n_embeddings, context_length, dropout)
        self.feed_forward = FeedForward(n_embeddings, dropout)
        self.layer_normalization_self_attention = nn.LayerNorm(n_embeddings)
        self.layer_normalization_feed_forward = nn.LayerNorm(n_embeddings)
        
        
    def forward(self, x):
        x = x + self.self_attention(self.layer_normalization_self_attention(x))
        return x + self.feed_forward(self.layer_normalization_feed_forward(x))