import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.functional as F
class TokenCompressor(nn.Module):
    def __init__(self, top_k=64):
        super().__init__()
        self.top_k = top_k
        self.attn_proj = nn.Linear(32, 1)  # Token dim=32

    def forward(self, tokens, context=None):
        # tokens: (B, N_tokens, C), context: (B, C) or None
        # Attention-guided: use context (e.g., text embedding) if provided
        if context is not None:
            # Compute attention as dot product with context
            attn_scores = (tokens * context.unsqueeze(1)).sum(-1)  # (B, N_tokens)
        else:
            attn_scores = self.attn_proj(tokens).squeeze(-1)  # (B, N_tokens)
        attn_scores = F.softmax(attn_scores, dim=-1)
        topk_scores, idx = torch.topk(attn_scores, self.top_k, dim=1)
        batch_indices = torch.arange(tokens.size(0)).unsqueeze(-1).expand(-1, self.top_k)
        compressed = tokens[batch_indices, idx]  # (B, top_k, C)
        return compressed
