"""Low-level gradient / force utilities shared by the gravitational message functions."""

from __future__ import annotations

import torch

__all__ = ["soften_distance_sq", "inverse_square_acceleration", "compute_acceleration"]


def soften_distance_sq(
    xi: torch.Tensor, xj: torch.Tensor, softening: float = 1e-4
) -> torch.Tensor:
    """Squared distance with a gravitational softening parameter ``eps``."""
    diff = xj - xi  # (..., 3)
    return (diff * diff).sum(-1) + softening**2


def inverse_square_acceleration(
    delta_pos: torch.Tensor, dist_sq: torch.Tensor
) -> torch.Tensor:
    """Return the (dimensionless) inverse-square acceleration vector per pair.

    ``delta_pos`` is ``(..., 3)`` = ``x_j - x_i``; returns ``delta_pos / |d|^3``.
    """
    inv_r3 = dist_sq.rsqrt() ** 3
    return delta_pos * inv_r3.unsqueeze(-1)


def compute_acceleration(
    messages: torch.Tensor, edge_index: torch.Tensor, n_nodes: int
) -> torch.Tensor:
    """Aggregate per-edge acceleration vectors into per-node net acceleration.

    ``messages`` is ``(E, 3)``; ``edge_index`` is ``(2, E)`` COO. Returns ``(N, 3)``.
    """
    src, dst = edge_index[0], edge_index[1]
    acc = torch.zeros(n_nodes, messages.shape[1], device=messages.device)
    acc = acc.index_add_(0, dst, messages)
    return acc
