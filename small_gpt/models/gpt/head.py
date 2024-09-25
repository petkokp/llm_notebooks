import torch
import torch.nn.functional as F
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, head_size, n_embeddings, context_length, dropout):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, context_length, channel = x.shape
        k = self.key(x) # (batch_size, context_length, channel)
        q = self.query(x) # (batch_size, context_length, channel)
        # attention scores ("affinities")
        weighted_aggregation = q @ k.transpose(-2,-1) * channel**-0.5 # (batch_size, context_length, channel) @ (batch_size, channel, context_length) -> (batch_size, context_length, context_length)
        weighted_aggregation = weighted_aggregation.masked_fill(self.tril[:context_length, :context_length] == 0, float('-inf')) # (batch_size, context_length, context_length)
        weighted_aggregation = F.softmax(weighted_aggregation, dim=-1) # (batch_size, context_length, context_length)
        weighted_aggregation = self.dropout(weighted_aggregation)
        # weighted aggregation of the values
        v = self.value(x) # (batch_size, context_length, channel)
        return weighted_aggregation @ v # (batch_size, context_length, context_length) @  (batch_size, context_length, channel) ->  (batch_size, context_length, channel)