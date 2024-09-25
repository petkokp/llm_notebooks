import torch
import torch.nn as nn
from .head import Head

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embeddings, context_length, dropout):
        super(
            ).__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embeddings, context_length, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.projection(torch.cat([h(x) for h in self.heads], dim=-1)))