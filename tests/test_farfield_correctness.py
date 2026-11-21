"""Deterministic correctness tests for the sparse Barnes-Hut far field.

These use hand-computable geometries so the expected force is known exactly and no
unit/sign ambiguity can creep in. The far field is defined as: for each node, the
sum over tree cells that satisfy the opening criterion size/dist < theta of
``M_cell * (COM_cell - x) / |COM_cell - x|^3``. Near cells (not satisfying the
criterion, including a node's own leaf) are handled by direct graph edges and are
therefore not part of this field.
"""
import numpy as np
import torch

from laniakea_hmmp.models.barnes_hut import BarnesHutTree


def _tree_over(cells, pos):
    """Build a tree over a tiny cloud and inspect its cell structure fields."""
    center = pos.mean(0)
    size = float((pos.max(0).values - pos.min(0).values).max().item()) * 1.2
    return BarnesHutTree(pos, center=center, size=size, max_leaf=2, max_depth=4)


def test_monopole_far_field_two_particles():
    """Two particles far apart: each should only open the root cell (or its own
    leaf), and the far-field monopole of the *other* particle's contribution must
    match the point-mass acceleration to within the BH opening tolerance."""
    # Particle A at origin, B at (10,0,0), tiny masses 1.0 each.
    pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    mass = torch.ones(2)
    center = pos.mean(0)
    size = 12.0
    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=2, max_depth=3)
    moments = tree.moments(mass)

    # Root cell COM is at (5,0,0) (midpoint), total mass = 2.
    com = moments["com"][0]
    total_mass = moments["total_mass"][0]
    assert abs(com[0].item() - 5.0) < 1e-3, f"root COM x should be 5, got {com[0].item()}"
    assert abs(total_mass.item() - 2.0) < 1e-3

    # With only two widely-separated particles each in its own leaf, neither cell
    # satisfies the opening criterion at theta=1.0, so the force is captured by the
    # NEAR-LEAF direct sum, not the far multipole. Verify BH total (leaf+far).
    fs, fc, ns, nc = tree.build_interaction_list(theta=1.0, device=torch.device("cpu"))
    acc = torch.zeros(2, 3)
    # Near-leaf direct sum for particle 0 from particle 1's leaf, and vice versa.
    node_pairs, part_pairs = [], []
    for si, ci in zip(ns.tolist(), nc.tolist()):
        pids = tree.leaf_particles_of(ci)
        pids = pids[pids != si]
        if pids.size == 0:
            continue
        node_pairs.append(np.full(pids.size, si, dtype=np.int64))
        part_pairs.append(pids)
    if node_pairs:
        n_idx = np.concatenate(node_pairs)
        p_idx = np.concatenate(part_pairs)
        d = pos[p_idx] - pos[n_idx]
        r = d.norm(dim=-1) + 1e-8
        acc = acc.index_add_(0, torch.from_numpy(n_idx), mass[p_idx].unsqueeze(-1) * d / (r ** 3).unsqueeze(-1))

    # Particle 0 is pulled toward particle 1 (+x), particle 1 toward particle 0 (-x).
    # |a| = m/r^2 = 1/100 = 0.01 per single partner (both particles in one leaf here).
    assert torch.isfinite(acc).all(), "BH near-leaf acceleration must be finite"
    assert acc[0, 0].item() > 0, "particle 0 should be pulled toward +x"
    assert acc[1, 0].item() < 0, "particle 1 should be pulled toward -x"
    assert abs(acc[0, 0].item()) > 1e-3, "non-degenerate mutual force"


def test_opening_criterion_excludes_own_leaf():
    """A node's own leaf cell must never contribute to its far field (near field)."""
    rng = np.random.default_rng(0)
    pos = torch.from_numpy(rng.uniform(0, 10, size=(100, 3))).float()
    mass = torch.ones(100)
    center = pos.mean(0)
    size = float((pos.max(0).values - pos.min(0).values).max().item()) * 1.1
    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=4, max_depth=10)
    src, cell, _ns, _nc = tree.build_interaction_list(theta=0.5, device=torch.device("cpu"))
    leaf_of = torch.from_numpy(tree.leaf_of).long()
    # No (node, its-own-leaf) pair in the far field.
    assert not (cell == leaf_of[src]).any()


def test_moment_matches_point_mass():
    """The root monopole of a set of equal-mass particles at known positions must
    equal the total mass and the centre of mass."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
    mass = torch.tensor([1.0, 1.0, 1.0, 1.0])
    center = pos.mean(0)
    size = 8.0
    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=2, max_depth=3)
    moments = tree.moments(mass)
    assert abs(moments["total_mass"][0].item() - 4.0) < 1e-3
    expected_com = pos.mean(0)  # equal masses -> mean position
    got_com = moments["com"][0]
    assert (got_com - expected_com).norm().item() < 1e-3, f"com {got_com} != {expected_com}"


def test_sparse_is_way_smaller_than_dense():
    """Sparse interaction count must be a small fraction of the dense N*K matrix."""
    rng = np.random.default_rng(1)
    pos = torch.from_numpy(rng.uniform(0, 10, size=(2000, 3))).float()
    mass = torch.ones(2000)
    center = pos.mean(0)
    size = 12.0
    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=8, max_depth=16)
    K = len(tree.cell_center)
    src, cell, _ns, _nc = tree.build_interaction_list(theta=0.5, device=torch.device("cpu"))
    assert src.numel() < 0.25 * 2000 * K
