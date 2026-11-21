"""Gravitational message functions.

Implements the manuscript's Newtonian message (Eq. 1) as a learnable inductive
bias: by default the message carries the inverse-square acceleration vector, which
is then mixed with the source node's learned features by a small MLP. ``use_learned``
controls whether the pure gravitational inductive bias is preserved or a learned
residual is added on top.
"""

from __future__ import annotations

import torch
from torch import nn

from .grad_utils import soften_distance_sq

__all__ = ["GravitationalMessage"]


class GravitationalMessage(nn.Module):
    """Differentiable Newtonian-gravity message function.

    For an edge ``(i <- j)`` the base message is
        ``m_ij = G * m_i * m_j * (x_j - x_i) / |x_j - x_i + eps|^3``,
    i.e. the gravitational force on particle ``i`` from ``j``. An optional learned
    MLP mixes this geometric bias with node features to correct softening and
    other N-body systematics.
    """

    def __init__(
        self,
        node_dim: int = 2,
        hidden_dim: int = 64,
        use_learned: bool = True,
        softening: float = 1e-4,
    ):
        super().__init__()
        self.softening = softening
        self.use_learned = use_learned
        self.node_dim = node_dim

        if use_learned:
            # Fold G * m_j into an edge weight, then mix force direction + features.
            self.edge_mlp = nn.Sequential(
                nn.Linear(3 + 2 * node_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 3),
            )
        else:
            self.edge_mlp = None

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mass: torch.Tensor,
        edge_index: torch.Tensor,
        edge_chunk: int | None = None,
    ) -> torch.Tensor:
        """Return per-edge messages of shape ``(E, 3)``.

        ``edge_chunk`` bounds how many edges are processed at once, keeping the peak
        transient tensor bounded for very large graphs (e.g. millions of nodes).
        """
        src, dst = edge_index[0], edge_index[1]
        e = src.numel()

        if edge_chunk is None or edge_chunk >= e or not self.use_learned:
            return self._forward_chunk(x, pos, mass, src, dst)

        outs = []
        for b in range(0, e, edge_chunk):
            si = src[b : b + edge_chunk]
            di = dst[b : b + edge_chunk]
            outs.append(self._forward_chunk(x, pos, mass, si, di))
        return torch.cat(outs, dim=0)

    def _forward_chunk(self, x, pos, mass, src, dst):
        """Message computation over a (sub)set of edges ``(src, dst)``."""
        xi = pos[src]
        xj = pos[dst]
        mi = mass[src]
        mj = mass[dst]

        dist_sq = soften_distance_sq(xi, xj, self.softening)
        delta = xj - xi
        base_dir = delta * (dist_sq.rsqrt() ** 3).unsqueeze(-1)  # (E,3) field direction
        strength = mi.unsqueeze(1) * mj.unsqueeze(1)  # (E,1) m_i m_j
        base = strength * base_dir  # (E,3)  ~ G m_i m_j * (x_j-x_i)/|d|^3

        if not self.use_learned:
            return base

        # Learned correction on top of the physical inductive bias.
        feat = torch.cat([base, x[src], x[dst]], dim=-1)  # (E, 3 + 2*node_dim)
        correction = self.edge_mlp(feat)
        return base + correction
