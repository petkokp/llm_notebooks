import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embeddings, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        return self.network(x)