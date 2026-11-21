"""Brute-force exact acceleration and tidal tensor -- the reference the operator is scored against.

The pair of quantities, in the repository's sign convention (``d = x_j - x_i``, so particle
``i`` is pulled *toward* particle ``j``):

    a_i = sum_j m_j d / r^3
    T_i = d a_i / d x_i = sum_j m_j [ 3 d d^T / r^5 - I / r^3 ]

with ``r^2 = |d|^2 + eps^2``. ``T`` is the tidal tensor (minus the Hessian of the potential,
and the object a vector-Jacobian product differentiates), which is exactly what the far-field
multipole approximation has to reproduce.

Why this module exists
----------------------
The same expression had been copy-pasted into several scripts with a **missing factor of
``r^3`` on the first term** -- written as ``3 d d^T / r^2 - I / r^3``, which is dimensionally
inconsistent and is *not* the Jacobian of the force. It happens to be right only at
``r = 1``. See ``docs/PROVENANCE.md`` §3.4; ``scripts/diagnose_tidal_convention.py`` shows the
two expressions against ``torch.autograd``. Every consumer should call this module instead of
re-deriving the formula, and ``tests/test_tidal_tensor.py`` pins the Jacobian identity so the
defect cannot come back through a refactor.

Cost. Only the requested ``query`` nodes are evaluated, so the reference is ``O(Q * N)`` rather
than ``O(N^2)``: that is what makes an exact comparison affordable on the 2.3M data. It is
accumulated in a compact six-component form (no ``(Q, N, 3, 3)`` intermediate), which keeps the
memory of a multi-million-``N`` run bounded by ``chunk``.
"""

from __future__ import annotations

import torch

__all__ = ["exact_force_and_tidal", "exact_tidal"]


def exact_force_and_tidal(
    pos: torch.Tensor,
    mass: torch.Tensor,
    query: torch.Tensor | None = None,
    softening: float = 1e-4,
    chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact pairwise ``(force, tidal)`` at ``query`` nodes.

    Args:
        pos: ``(N, 3)`` positions.
        mass: ``(N,)`` masses.
        query: node indices to evaluate; defaults to every node.
        softening: Plummer softening length ``eps`` (added to ``r^2``).
        chunk: query rows per pass. ``None`` picks a size that bounds the ``(chunk, N, 3)``
            intermediate to roughly 1e7 elements.

    Returns:
        ``(force, tidal)`` with shapes ``(Q, 3)`` and ``(Q, 3, 3)``. The self-interaction is
        masked: it would contribute ``-I/eps^3`` to the tidal tensor and dominate its norm.
    """
    n = pos.shape[0]
    if query is None:
        query = torch.arange(n, device=pos.device)
    query = query.reshape(-1)
    q = int(query.numel())
    if chunk is None:
        chunk = max(1, int(1.0e7 // max(n, 1)))
    chunk = max(1, min(int(chunk), q))

    dtype = pos.dtype
    device = pos.device
    force = torch.zeros(q, 3, dtype=dtype, device=device)
    tidal = torch.zeros(q, 3, 3, dtype=dtype, device=device)
    mass_row = mass.reshape(-1).to(dtype).unsqueeze(0)  # (1, N)

    for s in range(0, q, chunk):
        e = min(s + chunk, q)
        qi = query[s:e]
        d = pos.unsqueeze(0) - pos[qi].unsqueeze(1)          # (c, N, 3) = x_j - x_i
        r2 = (d * d).sum(-1) + softening ** 2                # (c, N)
        # Mask self: r2 -> inf makes every 1/r^k term vanish (and d is already 0 there).
        r2[torch.arange(e - s, device=device), qi] = float("inf")
        inv_r3 = r2.rsqrt() ** 3
        w = mass_row * inv_r3                                # m_j / r^3
        force[s:e] = torch.einsum("cn,cnI->cI", w, d)

        # Six independent components of sum_j m_j [3 d d^T / r^5 - I / r^3].
        u = 3.0 * w / r2                                      # 3 m_j / r^5
        iso = w.sum(1)                                       # sum_j m_j / r^3
        dx, dy, dz = d.unbind(-1)
        tidal[s:e, 0, 0] = torch.einsum("cn,cn->c", u, dx * dx) - iso
        tidal[s:e, 1, 1] = torch.einsum("cn,cn->c", u, dy * dy) - iso
        tidal[s:e, 2, 2] = torch.einsum("cn,cn->c", u, dz * dz) - iso
        tidal[s:e, 0, 1] = tidal[s:e, 1, 0] = torch.einsum("cn,cn->c", u, dx * dy)
        tidal[s:e, 0, 2] = tidal[s:e, 2, 0] = torch.einsum("cn,cn->c", u, dx * dz)
        tidal[s:e, 1, 2] = tidal[s:e, 2, 1] = torch.einsum("cn,cn->c", u, dy * dz)

    return force, tidal


def exact_tidal(*args, **kwargs) -> torch.Tensor:
    """The tidal tensor alone (see :func:`exact_force_and_tidal`)."""
    return exact_force_and_tidal(*args, **kwargs)[1]
