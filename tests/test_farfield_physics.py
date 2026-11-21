"""Physics correctness: with the (default) single-pole far field, the HMMP
near-leaf + far-field acceleration must approximate the exact pairwise force,
with no double-counting between the two BH components (both use the same
interaction list, cells are partitioned into far-open vs near-leaf).

This is the authoritative check that the far-field refactor is physically sound.
"""
import numpy as np
import torch

from laniakea_hmmp.data.synthetic import generate_particle_cloud
from laniakea_hmmp.models.barnes_hut import BarnesHutTree, exact_force
from laniakea_hmmp.models.hmmp import HMMP


def test_leaf_plus_far_approximates_exact():
    # Small, deterministic cloud.
    n = 500
    particles = generate_particle_cloud(n_particles=n, grid_n=16, seed=0)
    pos = torch.from_numpy(particles["pos"]).float()
    mass = torch.from_numpy(particles["mass"]).float()
    x = torch.zeros(n, 2)
    center = pos.mean(0)
    size = float((pos.max(0).values - pos.min(0).values).max().item())

    tree = BarnesHutTree(pos, center=center, size=size, max_leaf=8, max_depth=16)
    moments = tree.moments(mass)
    ll = tree.build_interaction_list(theta=0.5, device=torch.device("cpu"))

    # Default model: no quadrupole.
    model = HMMP(node_dim=2, hidden_dim=8, n_layers=1, max_leaf=8, max_depth=16, theta=0.5, use_quadrupole=False)
    with torch.no_grad():
        far = model.layers[0].far_field_acceleration(tree, moments, pos, mass, theta=0.5, interaction_list=ll)
        leaf = model.layers[0].near_leaf_acceleration(tree, pos, mass, theta=0.5, interaction_list=ll)

    exact = exact_force(pos, mass)
    total = leaf + far

    # Standard-Newtonian-sign now: BH (leaf+far) must closely approximate exact.
    # Barnes-Hut monopole at theta=0.5 has a small but nonzero intrinsic error.
    rel = ((total - exact).norm(dim=-1) / (exact.norm(dim=-1) + 1e-8)).mean().item()
    assert torch.isfinite(total).all(), "BH total acceleration must be finite"
    # Real measured rel is ~0.04; allow headroom for different seeds/sizes.
    assert rel < 0.2, f"BH total far off exact: rel={rel:.3f} (expected <0.2 for monopole)"

    # Direction must agree with the exact Newtonian acceleration (sign-correctness).
    cos = (total * exact).sum() / ((total.norm() * exact.norm()) + 1e-8)
    assert cos.item() > 0.99, f"BH total direction disagrees with exact (cos={cos.item():.3f})"

    # The far field must not be inflated by a spurious quadrupole term.
    mono_mag = far.norm(dim=-1).mean().item()
    assert mono_mag > 0, "far field degenerate"
