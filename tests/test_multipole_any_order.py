"""Regression guard for the order-general far field.

Why this exists
---------------
``models/multipole_any_order.py`` was committed with an unterminated module docstring: an edit that
rewrote the STATUS block left the original docstring's closing delimiter behind, so the file held a
stray ``\"\"\"`` and did not parse at all. **Nothing caught it**, because no test imported the module
and the script that uses it had been run before the edit. The repository's suite stayed green while a
committed module was unimportable.

So this file does two things, and the first one is the important one: it *imports* the module, and it
pins the far field against a longhand monopole sum, which is the check that originally found the
5.0x multiplicity defect in ``_cell_rows``.
"""
import numpy as np
import pytest
import torch

from laniakea_hmmp.models.barnes_hut import BarnesHutTree
from laniakea_hmmp.models.multipole_any_order import far_field, near_field


def _cloud(n=300, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3)) * 100.0
    mass = np.ones(n)
    return pos, mass


def test_module_imports_and_exposes_api():
    """The cheapest possible guard, and the one that was missing."""
    from laniakea_hmmp.models import multipole_any_order as m

    for name in ("legendre_all", "far_field", "near_field", "complete_field"):
        assert hasattr(m, name), name


def test_far_field_agrees_with_longhand_monopole():
    """l=0 must equal sum_cells -M_c d_c / r_c^3 to roundoff, on the SAME list.

    This is the check that exposed the 5.0x defect: ``_cell_rows`` expanded each unique cell once
    and credited it to a single requesting query, so a cell admitted by N nodes contributed once.
    A constant factor in the norm is the signature; a physics error would not be constant.
    """
    pos_np, mass_np = _cloud()
    n = pos_np.shape[0]
    pos32 = torch.tensor(pos_np, dtype=torch.float32)
    mass32 = torch.tensor(mass_np, dtype=torch.float32)
    pos = torch.tensor(pos_np, dtype=torch.float64)
    mass = torch.tensor(mass_np, dtype=torch.float64)

    tree = BarnesHutTree(pos32, max_leaf=16, max_depth=16)
    moments32 = tree.moments(mass32)
    com64 = tree.moments(mass32)["com"].double()

    qsel = torch.tensor(sorted(np.random.default_rng(1).choice(n, 24, replace=False)),
                        dtype=torch.long)
    theta = 0.5
    fs, fc, _ns, _nc = tree.build_interaction_list(
        theta=theta, device=torch.device("cpu"), query_nodes=qsel, opening_mode="com",
        com=com64)
    a_mine, _T = far_field(tree, pos, mass, qsel, theta, 0, com64, fs, fc)

    slot = {int(v): i for i, v in enumerate(qsel.tolist())}
    a_ref = np.zeros((qsel.numel(), 3))
    com_np = com64.numpy()
    pos64 = pos.numpy()
    for k in range(int(fs.numel())):
        qi, ci = int(fs[k]), int(fc[k])
        pids = np.asarray(tree._node_particles[ci])
        m_tot = float(mass.numpy()[pids].sum())
        d = pos64[qi] - com_np[ci]
        r = float(np.linalg.norm(d))
        a_ref[slot[qi]] += -m_tot * d / r ** 3

    rel = float((a_mine.numpy() - a_ref).__abs__().max() / max(np.abs(a_ref).max(), 1e-300))
    assert rel < 1e-10, f"far field disagrees with the longhand monopole sum: {rel:.3e}"


def test_near_field_matches_a_direct_sum():
    """Pure near field (theta -> 0, so nothing is accepted) against an explicit sum."""
    pos_np, mass_np = _cloud(n=200, seed=3)
    n = pos_np.shape[0]
    pos32 = torch.tensor(pos_np, dtype=torch.float32)
    mass32 = torch.tensor(mass_np, dtype=torch.float32)
    pos = torch.tensor(pos_np, dtype=torch.float64)
    mass = torch.tensor(mass_np, dtype=torch.float64)

    tree = BarnesHutTree(pos32, max_leaf=16, max_depth=16)
    com64 = tree.moments(mass32)["com"].double()
    qsel = torch.tensor([0, 7, 42, 199], dtype=torch.long)
    fs, fc, ns, nc = tree.build_interaction_list(
        theta=1e-6, device=torch.device("cpu"), query_nodes=qsel, opening_mode="com", com=com64)
    assert int(fs.numel()) == 0

    a_near, _T = near_field(tree, pos, mass, qsel, ns, nc, softening=1e-4)

    slot = {int(v): i for i, v in enumerate(qsel.tolist())}
    eps2 = 1e-4 ** 2
    pos64, mass64 = pos.numpy(), mass.numpy()
    a_ref = np.zeros((qsel.numel(), 3))
    for k in range(int(ns.numel())):
        qi, ci = int(ns[k]), int(nc[k])
        for j in np.asarray(tree._node_particles[ci]):
            if j == qi:
                continue
            d = pos64[j] - pos64[qi]
            r2 = float((d * d).sum()) + eps2
            a_ref[slot[qi]] += mass64[j] * d * r2 ** -1.5

    rel = float((a_near.numpy() - a_ref).__abs__().max() / max(np.abs(a_ref).max(), 1e-300))
    assert rel < 1e-12, f"near field disagrees with the direct sum: {rel:.3e}"
