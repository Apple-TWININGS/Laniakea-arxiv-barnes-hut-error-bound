"""Order-general truncated multipole far field, for the self-certifying surplus.

STATUS: VERIFIED as a far-field evaluator; the surplus it enables still does NOT work at operator
level. Both statements are measured.
--------------------------------------------------------------------------------------------
VERIFIED. ``scripts/_diag_any_order.py`` isolates the two halves and both now agree:
    pure far  (theta = 1e-6 so nothing is accepted) : near_field vs shipped near-only 7.3e-07
    pure far  at l = 0, same list, vs longhand      : 1.7e-16  (machine precision)
    complete  at l = 2, same list, vs shipped        : force 5.5e-05, tidal 6.9e-05
                                                       (limited by float32 in the shipped path)
The first version was wrong by exactly 5.0x in norm because ``_cell_rows`` expanded each UNIQUE
cell once and then mapped rows back through ``inv``, which can credit only one of the requests
sharing a cell -- so a cell accepted by N queries contributed to one of them. See that function's
docstring; the constant factor is the give-away for a dropped-multiplicity bug.

STILL FAILS. The hierarchical surplus is an essentially exact error estimator for an ISOLATED
cell (``scripts/selfcertify_surplus.py``: |err|/|surplus| median 1.008, sign alignment 0.998) but
it does not survive assembly. On a real tree with the interaction list held fixed
(``scripts/selfcertify_order_operator.py``, orders 0/2/4/6, two seeds, 64 query nodes):
    per-node |err|/|surplus|      median 0.27 and 0.15, ranges spanning 8-10 decades
    per-node -cos(err, surplus)   median -0.12 and -0.25   (near-orthogonal, not anti-parallel)
    per-node bar calibration      median 0.71 and 0.34, worst 3.5e7
i.e. the reported error bar UNDERSTATES the true error, sometimes by seven orders of magnitude.
Refining theta instead of the order fails the same way (1.57x-9.42x understatement). The reason is
structural: the assembled far-field error is a mixture over many accepted cells with different
convergence ratios and largely uncorrelated error directions, so one surplus vector cannot
represent it, and an unlucky borderline cell produces a large true error that the surplus does not
flag. This is "a mean is not a bound" one level up -- an error ESTIMATOR is not a bound either.

Therefore the deliverable of the paper is the provable per-interaction bound
(Appendix ``app:secondorder``), NOT a self-certified bar.

Why this exists at all
----------------------
``models/hmmp.py`` evaluates the far field at multipole order ``l <= 2`` only (``use_quadrupole``
is a boolean), so the hierarchical-surplus iterator -- which needs the SAME interaction list at two
orders, e.g. 0 and 2, or 2 and 4 -- cannot run on the shipped path. This module was written to
supply the missing far field for any even order, plus the exact near field.

What it computes
----------------
With d = x_query - com_cell, rel_b = x_b - com_cell and r = |d|, the Laplace expansion of the
potential is

    Phi(d) = sum_{l=0}^{L} r^{-(l+1)} sum_b m_b |rel_b|^l P_l(dhat . relhat_b)

which is exact for |rel_b| < r and is the same object ``paper.tex`` calls F_l. The acceleration and
the tidal tensor are obtained by differentiating that scalar with respect to d:

    a = grad_d Phi            (repo convention: a_i = sum_j m_j (x_j - x_i)/|x_j - x_i|^3)
    T = grad_d grad_d Phi

using ``torch.autograd`` rather than a hand-derived formula. That is deliberate: this repository has
already shipped one wrong hand-derived tidal term (defect D6) and one wrong hand-derived quadrupole
(D2), and the differentiated scalar cannot disagree with itself.

The far field is evaluated per interaction with the source cell expanded to its particles, so the
cost is O(sum_c n_c) per interaction rather than O(1). Making it O(1) per interaction requires
precomputing per-cell moments in a Cartesian multi-index basis plus the corresponding
kernel-derivative tensors, which is not done here.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

__all__ = ["legendre_all", "far_field", "near_field", "complete_field"]


def legendre_all(x: torch.Tensor, l_max: int) -> list[torch.Tensor]:
    """P_0(x) .. P_lmax(x) by the three-term recurrence; differentiable in x."""
    out = [torch.ones_like(x)]
    if l_max == 0:
        return out
    out.append(x)
    for l in range(1, l_max):
        out.append(((2 * l + 1) * x * out[l] - l * out[l - 1]) / (l + 1))
    return out


def _cell_rows(tree, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flat (owner, pid) expansion of arbitrary cells, NOT only leaves.

    ``BarnesHutTree.expand_leaf_particles`` covers leaves only (a non-leaf has zero leaf count),
    but admitted far cells are usually internal nodes, so this walks ``_node_particles`` instead.

    Rows are expanded **per request**, not per unique cell. The first version expanded each unique
    cell once and then mapped rows back through ``inv``, which can only credit ONE of the requests
    that share a cell -- so a cell accepted by N queries contributed to just one of them and the
    far field came out low by the mean cell-reuse factor. Measured on a cross-check at l=0: the
    far field was exactly 5.0x too small in norm (8.66e-2 against a longhand 4.33e-1), a constant
    factor, which is the signature of a dropped-multiplicity bug rather than a physics error. See
    ``scripts/_diag_any_order.py``, whose pure-near case (theta = 1e-6, nothing accepted) isolated
    the near field as correct to 7e-7 and left only this.
    """
    cells = np.asarray(cells, dtype=np.int64)
    uniq, inv = np.unique(cells, return_inverse=True)      # inv has one entry per REQUEST
    counts_u = np.array([tree._node_particles[int(u)].size for u in uniq], dtype=np.int64)
    starts_u = np.concatenate([[0], np.cumsum(counts_u)])
    if uniq.size:
        pid_u = np.concatenate([tree._node_particles[int(u)] for u in uniq])
    else:
        pid_u = np.empty(0, dtype=np.int64)
    req_counts = counts_u[inv]                             # particles per REQUEST
    total = int(req_counts.sum())
    owner = np.repeat(np.arange(len(cells)), req_counts)
    base = np.repeat(starts_u[inv], req_counts)
    within = np.arange(total) - np.repeat(np.cumsum(req_counts) - req_counts, req_counts)
    pid = pid_u[base + within]
    return owner, pid


def far_field(tree, pos: torch.Tensor, mass: torch.Tensor, query: torch.Tensor,
              theta: float, l_max: int, com: torch.Tensor,
              far_src: torch.Tensor, far_cell: torch.Tensor,
              chunk_interactions: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncated far field at ``query`` nodes, order ``l_max``, on a caller-supplied list.

    ``far_src``/``far_cell`` are the (query-node, accepted-cell) pairs of
    ``BarnesHutTree.build_interaction_list``; passing them in is what lets two calls at different
    ``l_max`` see the *same* list, which is the whole point of the surplus.
    """
    dev = pos.device
    n_q = query.numel()
    s = far_src.to(dev).long()
    c = far_cell.to(dev).long()
    owner, pid = _cell_rows(tree, c.detach().cpu().numpy())
    owner_t = torch.from_numpy(owner).to(dev).long()
    pid_t = torch.from_numpy(pid).to(dev).long()

    qpos = pos[query.long()]                                  # (n_q,3)
    comd = com.to(dev).double()
    ccom = comd[c]                                            # (M,3)
    # far_src entries are GLOBAL particle indices; map them to their slot in the query set
    slot = torch.full((pos.shape[0],), -1, device=dev, dtype=torch.long)
    slot[query.long()] = torch.arange(n_q, device=dev)
    d = (pos[s].double() - ccom).detach().clone().requires_grad_(True)

    m_rows = mass.to(dev).double()[pid_t]                     # (R,)
    rel = pos[pid_t].double() - comd[c][owner_t]              # (R,3) central offsets
    ylen = torch.linalg.norm(rel, dim=1)
    safe = torch.clamp(ylen, min=1e-300)
    relhat = rel / safe[:, None]

    a_out = torch.zeros(n_q, 3, device=dev, dtype=torch.float64)
    T_out = torch.zeros(n_q, 3, 3, device=dev, dtype=torch.float64)

    M = int(s.numel())
    for b0 in range(0, M, chunk_interactions):
        b1 = min(M, b0 + chunk_interactions)
        idx = torch.nonzero((owner_t >= b0) & (owner_t < b1), as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        d_b = d[b0:b1]
        o = owner_t[idx] - b0
        d_row = d_b[o]
        r = torch.linalg.norm(d_row, dim=1)
        dhat = d_row / r[:, None]
        cosg = torch.clamp((relhat[idx] * dhat).sum(1), -1.0, 1.0)
        P = legendre_all(cosg, l_max)
        phi = torch.zeros(d_b.shape[0], dtype=torch.float64, device=dev)
        for l in range(l_max + 1):
            phi = phi.index_add(0, o, m_rows[idx] * (safe[idx] ** l) * P[l] / r ** (l + 1))
        (g,) = torch.autograd.grad(phi.sum(), d_b, create_graph=True)
        cols = []
        for k in range(3):
            (gk,) = torch.autograd.grad(g[:, k].sum(), d_b, retain_graph=True)
            cols.append(gk)
        a_out.index_add_(0, slot[s[b0:b1]], g.detach())
        T_out.index_add_(0, slot[s[b0:b1]], torch.stack(cols, dim=1).detach())
    return a_out, T_out


def near_field(tree, pos: torch.Tensor, mass: torch.Tensor, query: torch.Tensor,
               near_src: torch.Tensor, near_cell: torch.Tensor,
               softening: float = 0.0, chunk: int = 2_000_000
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact direct sums over the non-admitted near leaves. Order independent by construction."""
    dev = pos.device
    n_q = query.numel()
    ns = near_src.detach().cpu().numpy()
    nc = near_cell.detach().cpu().numpy()
    a_out = torch.zeros(n_q, 3, device=dev, dtype=torch.float64)
    T_out = torch.zeros(n_q, 3, 3, device=dev, dtype=torch.float64)
    if ns.size == 0:
        return a_out, T_out
    owner, pid, mask = tree.expand_leaf_particles(nc, self_particles=ns)
    # The mask excludes the query particle from its own leaf; dropping it injects a self-term of
    # magnitude ~1/eps^2 and swamps everything (measured: 1e13 on a 1e0 field). The shipped
    # operator applies it, so this must too or the cross-check is meaningless.
    owner, pid = owner[mask], pid[mask]
    owner_t = torch.from_numpy(owner).to(dev).long()
    pid_t = torch.from_numpy(pid).to(dev).long()
    # near_src entries are GLOBAL particle indices; map them to their slot in the query set
    slot = torch.full((pos.shape[0],), -1, device=dev, dtype=torch.long)
    slot[query.long()] = torch.arange(query.numel(), device=dev)
    m = mass.to(dev).double()[pid_t]
    src = pos[pid_t].double()
    eps2 = float(softening) ** 2
    for b0 in range(0, owner_t.numel(), chunk):
        b1 = min(owner_t.numel(), b0 + chunk)
        o = owner_t[b0:b1]
        dest = torch.from_numpy(ns[o]).to(dev).long()
        dd = src[b0:b1] - pos[dest].double()
        r2 = (dd * dd).sum(1) + eps2
        inv3 = r2 ** -1.5
        inv5 = inv3 / r2
        mb = m[b0:b1]
        a_out.index_add_(0, slot[dest], mb[:, None] * dd * inv3[:, None])
        Tb = mb[:, None, None] * (3.0 * (dd[:, :, None] * dd[:, None, :]) * inv5[:, None, None]
                                  - torch.eye(3, device=dev, dtype=torch.float64)
                                  * inv3[:, None, None])
        T_out.index_add_(0, slot[dest], Tb)
    return a_out, T_out


def complete_field(tree, pos: torch.Tensor, mass: torch.Tensor, query: torch.Tensor,
                   theta: float, l_max: int, moments: dict, opening_mode: str = "com",
                   softening: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete operator = exact near leaves + order-``l_max`` far field, at one theta."""
    fs, fc, ns, nc = tree.build_interaction_list(
        theta=theta, device=pos.device, query_nodes=query, opening_mode=opening_mode,
        com=moments["com"])
    a_f, T_f = far_field(tree, pos.double(), mass.double(), query, theta, l_max,
                         moments["com"].double(), fs, fc)
    a_n, T_n = near_field(tree, pos.double(), mass.double(), query, ns, nc, softening)
    return a_f + a_n, T_f + T_n
