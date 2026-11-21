"""Octree cell geometry: does ``cell_center`` actually describe the cell it labels?

Pins defects D8a and D8b and their fix (``docs/PROVENANCE.md``, section 3.5). The defining property
of an octree cell is that every member particle lies inside ``[centre - size/2, centre + size/2]``.
That holds independently of any multipole approximation, so it is a pure invariant.

The defects, both on the historical default path:

  * **D8a** -- ``_build_tree`` offset a child centre by ``sz/2`` where an octant centre is at
    ``sz/4``, so every non-root cell's recorded centre was a full cell side from its contents.
    97-99% of cells violated containment regardless of how the root was placed.
  * **D8b** -- callers centred the root on the positional *mean* while giving it the max *span* as
    a side (``exact_nbody_sampled.py``: ``center = pos.mean(0)``). On a non-uniform cloud the mean
    can sit several percent of the span off the bounding-box centre, so particles fell outside the
    root box itself.

The fix derives the root from the particle bounding box and uses octant centres. Both
``legacy_geometry=True`` reproductions are still asserted, so the defect cannot silently return.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from laniakea_hmmp.models.barnes_hut import BarnesHutTree


def _membership(tree) -> list[list[int]]:
    n_cells = len(tree.cell_center)
    member: list[list[int]] = [[] for _ in range(n_cells)]
    for node in range(n_cells):
        if tree.cell_is_leaf[node]:
            member[node] = [int(p) for p in tree.leaf_particles_of(node)]
    for depth in range(int(tree.cell_depth.max()), 0, -1):
        for node in np.nonzero(tree.cell_depth == depth)[0]:
            parent = tree.cell_parent[node]
            if parent >= 0:
                member[parent].extend(member[node])
    return member


def _containment(tree) -> tuple[float, int, int]:
    """(worst |x - centre| in half-sides, violating cells, cells checked)."""
    member = _membership(tree)
    half = tree.cell_size / 2.0
    worst, bad, checked = 0.0, 0, 0
    for node in range(len(tree.cell_center)):
        if not member[node]:
            continue
        checked += 1
        pts = tree.pos[np.asarray(member[node])]
        e = float(np.abs(pts - tree.cell_center[node]).max(axis=1).max() / max(half[node], 1e-30))
        worst = max(worst, e)
        if e > 1.0 + 1e-3:
            bad += 1
    return worst, bad, checked


def _root_excess(tree) -> float:
    return float(np.abs(tree.pos - tree.cell_center[0]).max() / (tree.cell_size[0] / 2.0))


def _build(positions: torch.Tensor, *, legacy: bool = False, mean_root: bool = False,
           **kw) -> BarnesHutTree:
    if legacy:
        span = float((positions.max(0).values - positions.min(0).values).max().item())
        center = (positions.mean(0) if mean_root
                  else (positions.max(0).values + positions.min(0).values) / 2.0).detach()
        return BarnesHutTree(positions, center=center, size=span, max_leaf=8, max_depth=20,
                             legacy_geometry=True)
    return BarnesHutTree(positions, max_leaf=8, max_depth=20, **kw)


@pytest.fixture()
def cloud() -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.random((1500, 3)).astype(np.float32) * 100.0)


@pytest.fixture()
def gaussian() -> torch.Tensor:
    """Non-uniform: the mean sits ~5.8% of the span off the bounding-box centre (D8b)."""
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.normal(0, 20, (1500, 3)).astype(np.float32))


@pytest.mark.parametrize("name", ["cloud", "gaussian"])
def test_default_geometry_is_self_consistent(request, name: str) -> None:
    """Fixed path: every cell brackets its own particles, on uniform and non-uniform clouds."""
    tree = _build(request.getfixturevalue(name))
    worst, bad, checked = _containment(tree)
    assert checked > 0
    assert bad == 0, f"{bad}/{checked} cells violate containment (worst {worst:.3f})"
    assert worst <= 1.0 + 1e-3


@pytest.mark.parametrize("name", ["cloud", "gaussian"])
def test_default_root_contains_every_particle(request, name: str) -> None:
    """D8b fix: the root is the bounding box, so nothing escapes it."""
    tree = _build(request.getfixturevalue(name))
    assert _root_excess(tree) <= 1.0 + 1e-3


def test_explicit_size_is_honoured_as_a_lower_bound(cloud: torch.Tensor) -> None:
    """A caller-supplied ``size`` still widens the root; it can never shrink it below the bbox."""
    span = float((cloud.max(0).values - cloud.min(0).values).max().item())
    big = _build(cloud, size=span * 4.0)
    assert big.size >= span * 4.0 - 1e-6
    assert _root_excess(big) <= 1.0 + 1e-3
    small = _build(cloud, size=span * 1e-6)
    assert small.size >= span - 1e-6
    assert _root_excess(small) <= 1.0 + 1e-3


def test_pad_widens_the_root(cloud: torch.Tensor) -> None:
    a = _build(cloud)
    b = _build(cloud, pad=1.05)
    assert b.size == pytest.approx(a.size * 1.05, rel=1e-9)
    assert _root_excess(b) <= 1.0 + 1e-3


def test_legacy_geometry_reproduces_d8a(cloud: torch.Tensor) -> None:
    """The legacy child offset must still be reproducible, and must still be broken.

    Asserting the defect rather than xfailing it: if a future change silently makes the legacy
    path correct, already-committed artifacts would stop being reproducible and this test should
    say so loudly.
    """
    worst, bad, checked = _containment(_build(cloud, legacy=True))
    assert checked > 0
    assert bad > 0.9 * checked, (
        f"D8a not reproduced: only {bad}/{checked} cells violate containment "
        f"(worst {worst:.3f}); the legacy offset should break ~99%"
    )
    assert worst > 1.9, f"D8a signature is a 2.0 half-side offset, got {worst:.3f}"


def test_legacy_geometry_reproduces_d8b(gaussian: torch.Tensor) -> None:
    """The legacy mean-centred root must still let particles escape the root box."""
    assert _root_excess(_build(gaussian, legacy=True, mean_root=True)) > 1.0 + 1e-3


def test_com_criterion_does_not_read_cell_center(cloud: torch.Tensor) -> None:
    """The 'com' opening mode must be immune to D8: it measures to moments['com'] instead."""
    mass = torch.ones(cloud.shape[0], dtype=torch.float32)
    q = torch.arange(64)
    for legacy in (False, True):
        tree = _build(cloud, legacy=legacy)
        moments = tree.moments(mass)
        fs, fc, ns, nc = tree.build_interaction_list(
            theta=0.5, query_nodes=q, opening_mode="com", com=moments["com"]
        )
        assert fs.numel() > 0


def test_no_cell_centre_is_outside_its_parent_box(cloud: torch.Tensor) -> None:
    """Stronger invariant: a child's box must lie inside its parent's box."""
    tree = _build(cloud)
    half = tree.cell_size / 2.0
    for node in range(1, len(tree.cell_center)):
        parent = tree.cell_parent[node]
        assert parent >= 0
        offset = np.abs(tree.cell_center[node] - tree.cell_center[parent]).max()
        assert offset + half[node] <= half[parent] + 1e-3, (
            f"cell {node} (depth {tree.cell_depth[node]}) escapes its parent box: "
            f"offset {offset:.4g} + half {half[node]:.4g} > parent half {half[parent]:.4g}"
        )
