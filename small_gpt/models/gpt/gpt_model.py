import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from .block import Block

class GPT(nn.Module):
    def __init__(self, vocabulary_size, n_embeddings, n_heads, context_length, n_layers, dropout, bias):
        super().__init__()
        
        self.transformer = nn.ModuleDict(dict(
            token_embedding_table = nn.Embedding(vocabulary_size, n_embeddings),
            position_embedding_table = nn.Embedding(context_length, n_embeddings), # positions at which the tokens occur
            drop = nn.Dropout(dropout),
            blocks = nn.ModuleList([Block(n_embeddings, n_heads, context_length, dropout) for _ in range(n_layers)]),
            layer_norm = nn.LayerNorm(n_embeddings, bias),
        ))
        self.lm_head = nn.Linear(n_embeddings, vocabulary_size, bias=False)
        self.transformer.token_embedding_table.weight = self.lm_head.weight # weight tying
        self.apply(self._init_weights)
        
        # scaled initialization to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
                
    def get_parameters_number(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        ###
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # return logits, loss
        ###
        
        device = idx.device
        
        _, context_length = idx.shape
        
        token_embedding = self.transformer.token_embedding_table(idx) # returns (batch, time, channel) = (batch_size, context_length, vocabulary_size)
        position_embedding = self.transformer.position_embedding_table(torch.arange(context_length, device=device)) # (context_size, channel)
        x = self.transformer.drop(token_embedding + position_embedding) # (batch_size, context_length, channel) - information for both tokens' identities and their positions
        
        for block in self.transformer.blocks:
            x = block(x)

        x = self.layer_norm(x)
        
        if targets is None:
            logits = self.lm_head(x[:, [-1], :]) # only forward lm_head on the very last position
            loss = None
        else:
            logits = self.lm_head(x) # (batch, context_length, vocabulary_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()} # all candidate parameters
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # remove parameters that don't require grad
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in no_decay_params)
        print(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Number of non-decayed parameter tensors: {len(no_decay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer
    
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