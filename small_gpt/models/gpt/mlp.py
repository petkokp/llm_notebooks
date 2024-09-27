import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_embeddings, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embeddings, 4 * n_embeddings, bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embeddings, n_embeddings, bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x