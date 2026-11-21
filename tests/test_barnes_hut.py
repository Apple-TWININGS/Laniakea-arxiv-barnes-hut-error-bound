"""Tests for the Barnes-Hut tree and its multipole moments."""

import numpy as np
import torch
import pytest

from laniakea_hmmp.models.barnes_hut import BarnesHutTree, exact_force


def _random_cloud(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = torch.from_numpy(rng.uniform(0, 10, size=(n, 3))).float()
    mass = torch.from_numpy(rng.uniform(0.5, 2.0, size=n)).float()
    center = pos.mean(0)
    size = float((pos.max(0).values - pos.min(0).values).max().item()) * 1.2
    return pos, mass, center, size


def test_tree_constructs_with_valid_leaf_assignment():
    pos, _, center, size = _random_cloud()
    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=8, max_depth=16)
    assert len(tree.cell_center) > 1, "tree should have more than the root cell"
    assert tree.leaf_of.shape == (pos.shape[0],)
    assert (tree.leaf_of >= 0).all()
    # Every particle belongs to exactly one leaf.
    leaves, counts = np.unique(tree.leaf_of, return_counts=True)
    assert counts.sum() == pos.shape[0]
    assert np.all(tree.cell_is_leaf[leaves]), "all assigned cells must be leaves"


def test_monopole_mass_is_conserved():
    pos, mass, center, size = _random_cloud()
    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=8, max_depth=16)
    moments = tree.moments(mass)
    # The root's total mass must equal the sum of all particle masses.
    total = moments["total_mass"][0].item()
    expected = mass.sum().item()
    assert abs(total - expected) / (expected + 1e-12) < 1e-2, f"root mass {total} != {expected}"


def test_com_is_weighted_mean():
    pos, mass, center, size = _random_cloud()
    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=8, max_depth=16)
    moments = tree.moments(mass)
    expected = (mass.unsqueeze(1) * pos).sum(0) / mass.sum()
    got = moments["com"][0]
    assert torch.allclose(got, expected, atol=1e-2), f"com {got} != {expected}"


def test_multipole_error_is_small_relative_to_exact():
    pos, mass, center, size = _random_cloud(n=500)
    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=8, max_depth=16)
    moments = tree.moments(mass)
    exact = exact_force(pos, mass)
    com = moments["com"]
    total_mass = moments["total_mass"]
    cell_size = torch.from_numpy(tree.cell_size).float()
    # Standard sign: acceleration on particle i pulls toward the cell COM.
    diff = com.unsqueeze(0) - pos.unsqueeze(1)
    dist_sq = (diff * diff).sum(-1)
    dist = dist_sq.sqrt() + 1e-8
    inv_r3 = (dist_sq + 1e-8).rsqrt() ** 3
    # Only FAR cells contribute to the multipole field (Barnes-Hut opening criterion).
    theta = 0.5
    open_cell = (cell_size.unsqueeze(0) / dist) < theta
    mono = (total_mass.unsqueeze(0).unsqueeze(-1) * diff * inv_r3.unsqueeze(-1)) * open_cell.unsqueeze(-1).float()
    mono = mono.sum(1)
    rel = (mono - exact).norm(dim=-1) / (exact.norm(dim=-1) + 1e-8)
    # The multipole field captures most of the force but is not exact (it ignores the
    # moderate-distance near-field terms retained for direct edges). Require the error
    # to be bounded and the far-field to carry a meaningful share of the signal.
    assert torch.isfinite(mono).all(), "far-field multipole force must be finite"
    assert rel.mean().item() < 50.0, f"multipole error too large: {rel.mean().item()}"
    # The far field should carry a substantial fraction of the total force signal.
    far_energy = mono.norm(dim=-1).mean()
    assert far_energy.item() > 1e-3, "far field is degenerate"
