import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from .block import Block

class GPT(nn.Module):
    def __init__(self, vocabulary_size, n_embeddings, n_heads, context_length, n_layers, dropout, bias):
        super().__init__()
        
        self.context_length = context_length
        
        self.transformer = nn.ModuleDict(dict(
            token_embedding_table = nn.Embedding(vocabulary_size, n_embeddings),
            position_embedding_table = nn.Embedding(context_length, n_embeddings), # positions at which the tokens occur
            drop = nn.Dropout(dropout),
            blocks = nn.ModuleList([Block(n_embeddings, bias, n_heads, dropout, context_length) for _ in range(n_layers)]),
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
        device = idx.device
        
        _, context_length = idx.shape
        
        token_embedding = self.transformer.token_embedding_table(idx) # returns (batch, time, channel) = (batch_size, context_length, vocabulary_size)
        position_embedding = self.transformer.position_embedding_table(torch.arange(context_length, device=device)) # (context_size, channel)
        x = self.transformer.drop(token_embedding + position_embedding) # (batch_size, context_length, channel) - information for both tokens' identities and their positions
        
        for block in self.transformer.blocks:
            x = block(x)

        x = self.transformer.layer_norm(x)
        
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
    
    def crop_context_length(self, context_length):
        assert context_length <= self.context_length
        self.context_length = context_length
        self.transformer.position_embedding_table.weight = nn.Parameter(self.transformer.position_embedding_table.weight[:context_length])
        for block in self.transformer.blocks:
            if hasattr(block.attention, 'bias'):
                block.attention.bias = block.attention.bias[:,:,:context_length,:context_length]
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:] # crop context at context_length
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # scale logits by temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probabilities = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1) # sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # append to sequence

        return idx
