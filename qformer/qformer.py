import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class QFormer(nn.Module):
    def __init__(self, num_queries=32):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, 32))
        self.cross_attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    def forward(self, image_tokens, text_queries):
        # image_tokens: (B, N, 32), text_queries: list of str
        text_inputs = self.tokenizer(text_queries, return_tensors='pt', padding=True, truncation=True)
        text_embeds = self.text_encoder(**text_inputs).last_hidden_state.mean(dim=1)  # (B, 768)
        queries = self.query_embed.unsqueeze(0).expand(image_tokens.size(0), -1, -1)  # (B, num_queries, 32)
        # Cross-attend queries to image tokens
        aligned, _ = self.cross_attn(queries, image_tokens, image_tokens)
        return aligned
