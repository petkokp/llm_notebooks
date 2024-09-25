import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Block

class GPT(nn.Module):
    def __init__(self, vocabulary_size, n_embeddings, n_heads, context_length, n_layers, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(context_length, n_embeddings) # positions at which the tokens occur
        self.blocks = nn.Sequential(*[Block(n_embeddings, n_heads, context_length, dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocabulary_size)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        _, context_length = idx.shape
        
        token_embedding = self.token_embedding_table(idx) # returns (batch, time, channel) = (batch_size, context_length, vocabulary_size)
        position_embedding = self.position_embedding_table(torch.arange(context_length, device=device)) # (context_size, channel)
        x = token_embedding + position_embedding # (batch_size, context_length, channel) - information for both tokens' identities and their positions                                                   
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x) # (batch, context_length, vocabulary_size)
        
        if targets is None:
            loss = None
        else:        
            batch, time, channel = logits.shape
            
            logits = logits.view(batch * time, channel)
            targets = targets.view(batch * time)
            
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        _, context_length = idx.shape
        # idx is (batch_size, context_length) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:] # crop context
            
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (batch_size, channel)

            probs = F.softmax(logits, dim=-1) # (batch_size, channel)
            
            next_idx = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            
            idx = torch.cat((idx, next_idx), dim=1) # (batch_size, context_length + 1)
        return idx