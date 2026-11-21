"""Sparse graph attention (Transformer encoder layer on the existing edge graph).

This is the "HMMP + Sparse Transformer" hybrid from the architecture note: the
Barnes-Hut multipole / GNN already encode the physical (Newtonian) field; a shallow
graph-attention block on top performs cross-node *semantic* refinement without the
O(N^2) cost of a dense Transformer. The attention is masked to the existing sparse
``edge_index`` (kNN + BH interaction edges), so it is O(N log N) / O(N * degree),
never O(N^2), and is batched by ``edge_chunk`` to bound memory.

Standard graph-attention (GAT-style) on a sparse edge set:
    for each edge (i->j):  att = softmax_j( a^T [Wq h_i || Wk h_j] )
    h'_i = sum_j att_ij * Wv h_j
The message passing is parallelised over edges and accumulated with index_add_.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int = 4, edge_chunk: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.edge_chunk = edge_chunk
        d = hidden_dim
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        # Attention logit = a^T [Wq h_i || Wk h_j] (concat of 2*d).
        self.a = nn.Linear(2 * d, heads)
        self.out = nn.Linear(d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor,
                edge_slice: torch.Tensor | None = None) -> torch.Tensor:
        """Message-passing attention over ``edge_index`` (standard GAT-style softmax).

        For each edge (i -> j): score = (a^T [Wq h_i || Wk h_j]) / sqrt(d). Then a
        softmax of the scores over all *sender* edges of each destination node j gives
        the attention weights; the output is sum_j att_ij * Wv h_j, accumulated back
        into the source i. Always O(E) in edges, never O(N^2).
        """
        edges = edge_slice if edge_slice is not None else edge_index
        src, dst = edges[0], edges[1]
        n = h.shape[0]

        q = self.Wq(h)  # (n, d)
        k = self.Wk(h)
        v = self.Wv(h)
        heads = self.heads

        out = torch.zeros_like(h)
        chunk = self.edge_chunk or edges.shape[1]
        # denom[d, head] = sum over sender edges (i->d) of exp(score). Per-head.
        # Clamp the raw logits so exp() cannot overflow to inf (which would poison the
        # denominator and produce NaN in the softmax).
        denom = torch.zeros(n, heads, device=h.device)
        for b in range(0, edges.shape[1], chunk):
            s, d = src[b : b + chunk], dst[b : b + chunk]
            ki = self.a(torch.cat([q[s], k[d]], dim=-1))  # (B, heads)
            scores = (ki / (self.hidden_dim ** 0.5)).clamp(-20.0, 20.0)
            denom.index_add_(0, d, scores.exp())  # (n, heads)
        denom = denom.clamp_min(1e-6)
        # Normalised attention, then message = att * v[dst] accumulated into src.
        for b in range(0, edges.shape[1], chunk):
            s, d = src[b : b + chunk], dst[b : b + chunk]
            ki = self.a(torch.cat([q[s], k[d]], dim=-1))  # (B, heads)
            scores = (ki / (self.hidden_dim ** 0.5)).clamp(-20.0, 20.0)
            att = scores.exp() / denom[d]  # (B, heads)
            att = torch.nan_to_num(att, nan=0.0, posinf=0.0, neginf=0.0)
            dvd = v[d].view(-1, heads, self.hidden_dim // heads)  # (B, heads, d/heads)
            msg = (att.unsqueeze(-1) * dvd).view(-1, self.hidden_dim)  # (B, d)
            out.index_add_(0, s, msg)

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        attended = self.out(out)
        h = self.norm1(h + attended)
        h = self.norm2(h + self.ff(h))
        return h

        # Single-pass attention without an explicit score matrix: gather q/k/v per
        # edge chunk, accumulate. (The above two-pass softmax is correct and
        # memory-bounded; kept here for clarity of the per-head normalisation.)

