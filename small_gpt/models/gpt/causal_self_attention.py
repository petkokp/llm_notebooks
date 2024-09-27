import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embeddings, n_heads, bias, dropout, context_length):
        super().__init__()
        assert n_embeddings % n_heads == 0
        self.c_attn = nn.Linear(n_embeddings, 3 * n_embeddings, bias) # key, query, value projections for all heads in a batch
        self.c_proj = nn.Linear(n_embeddings, n_embeddings, bias) # output projection
        self.attn_dropout = nn.Dropout(dropout) # regularization
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_heads
        self.n_embd = n_embeddings
        self.dropout = dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') # flash attention
        if not self.flash:
            print("WARNING - using slow attention because flash attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(context_length, context_length))
                                        .view(1, 1, context_length, context_length))

    def forward(self, x):
        batch_size, context_length, channel = x.size() # batch size, context length, embedding dimensionality

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, context_length, self.n_head, channel // self.n_head).transpose(1, 2) # (batch_size, nh, context_length, hs)
        q = q.view(batch_size, context_length, self.n_head, channel // self.n_head).transpose(1, 2) # (batch_size, nh, context_length, hs)
        v = v.view(batch_size, context_length, self.n_head, channel // self.n_head).transpose(1, 2) # (batch_size, nh, context_length, hs)

        # causal self-attention - (batch_size, nh, context_length, hs) x (batch_size, nh, hs, context_length) -> (batch_size, nh, context_length, context_length)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True) # efficient attention using Flash Attention CUDA kernels
        else:
            # slow attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:context_length,:context_length] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(batch_size, context_length, channel) # re-assemble all head outputs side by side

        return self.resid_dropout(self.c_proj(y)) # output projection