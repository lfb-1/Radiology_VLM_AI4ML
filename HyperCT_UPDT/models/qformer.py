"""
Q-Former Adapter for Vision-Language Alignment

Bridges pooled DINOv3 vision tokens (768-dim) to LLM hidden space (4096-dim).
Uses learnable query tokens with cross-attention to vision features, followed
by a linear projection to the LLM's input dimension.

Architecture:
    Pooled vision tokens (N, 768) → Cross-Attention with queries → MLP → (N_q, 4096)

Supports optional task conditioning: task embedding is added to queries
so the Q-Former can produce task-aware representations.
"""

import torch
import torch.nn as nn
from typing import Optional


class QFormerLayer(nn.Module):
    """Single Q-Former transformer layer: self-attn → cross-attn → FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Self-attention on queries
        q_normed = self.norm1(queries)
        sa_out, _ = self.self_attn(q_normed, q_normed, q_normed)
        queries = queries + sa_out

        # Cross-attention: queries attend to vision tokens
        q_normed = self.norm2(queries)
        kv_normed = self.norm_kv(kv)
        ca_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed)
        queries = queries + ca_out

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class QFormerAdapter(nn.Module):
    """
    Q-Former adapter that compresses vision tokens and projects to LLM space.

    Pipeline:
        vision_tokens (B, N_vision, 768)
        → Q-Former layers (cross-attention with learnable queries)
        → output projection (768 → 4096)
        → aligned_tokens (B, num_queries, 4096)
    """

    def __init__(self, vision_dim: int = 768, llm_dim: int = 4096,
                 num_queries: int = 64, num_layers: int = 6,
                 num_heads: int = 12, dropout: float = 0.1,
                 num_tasks: int = 0):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, vision_dim) * 0.02)

        self.layers = nn.ModuleList([
            QFormerLayer(vision_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(vision_dim)
        self.output_proj = nn.Linear(vision_dim, llm_dim)

        # Optional task conditioning
        self.task_conditioning = num_tasks > 0
        if self.task_conditioning:
            self.task_embeddings = nn.Embedding(num_tasks, vision_dim)

    def forward(self, vision_tokens: torch.Tensor,
                task_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            vision_tokens: (B, N_vision, vision_dim) pooled vision features
            task_id: (B,) optional task indices for conditioning
        Returns:
            aligned_tokens: (B, num_queries, llm_dim)
        """
        B = vision_tokens.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        if self.task_conditioning and task_id is not None:
            task_emb = self.task_embeddings(task_id).unsqueeze(1)  # (B, 1, D)
            queries = queries + task_emb

        for layer in self.layers:
            queries = layer(queries, vision_tokens)

        queries = self.output_norm(queries)
        return self.output_proj(queries)
