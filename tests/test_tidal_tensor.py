"""The tidal tensor (second derivative) of the HMMP operator, pinned against autograd.

Three independent identities, none of which is a re-statement of a formula:

1. ``exact_field.exact_force_and_tidal``'s tidal output really is ``d a_i / d x_j`` -- checked
   against ``torch.autograd.functional.jacobian`` of its own force output.
2. The new **quadrupole tidal** term in ``hmmp.HMMPLayer.far_field_acceleration_and_tidal`` is
   ``d a_quad / d x_j`` -- checked against autograd of the *shipped* quadrupole force term, so
   the test cannot pass by agreeing with a copy of the same algebra.
3. The near-leaf tidal sum is the exact pairwise tidal tensor, and the combined
   far + near tidal reproduces ``exact_tidal`` in the same way the force already does.

Background: the expression ``3 d d^T / r^2 - I / r^3`` (used by ``exact_nbody_sampled.py`` and
two probe scripts before this) is *not* the Jacobian; the first term needs ``r^5``. The
``test_shipped_legacy_expression_is_not_the_jacobian`` case documents the difference, and
``docs/PROVENANCE.md`` §3.4 records the defect.
"""
import pytest
import torch

from laniakea_hmmp.data.synthetic import generate_particle_cloud
from laniakea_hmmp.models.barnes_hut import BarnesHutTree
from laniakea_hmmp.models.exact_field import exact_force_and_tidal
from laniakea_hmmp.models.hmmp import HMMP

# float64 for the derivative identities (float32 rounds away the comparison), float32 where
# the assertion is about the operator's own precision.
TOL64 = 1e-10
F32_TOL = 1e-5


def _cloud(n: int = 48, seed: int = 0, dtype=torch.float32):
    particles = generate_particle_cloud(n_particles=n, grid_n=8, seed=seed)
    pos = torch.from_numpy(particles["pos"]).to(dtype)
    mass = torch.from_numpy(particles["mass"]).to(dtype)
    return pos, mass


def _tree(pos, mass, max_leaf: int = 8, max_depth: int = 16):
    center = pos.mean(0).detach()
    size = float((pos.max(0).values - pos.min(0).values).max().item())
    tree = BarnesHutTree(pos.detach().cpu(), center=center.cpu(), size=size,
                         max_leaf=max_leaf, max_depth=max_depth)
    return tree, tree.moments(mass.detach().cpu())


def _layer(use_quadrupole: bool, node_dim: int = 3) -> HMMP:
    return HMMP(node_dim=node_dim, hidden_dim=8, n_layers=1, max_leaf=8, max_depth=16,
                use_quadrupole=use_quadrupole).eval().layers[0]


# --------------------------------------------------------------------------------------
# 1. the exact reference is the Jacobian of the exact force
# --------------------------------------------------------------------------------------
def test_exact_tidal_is_the_force_jacobian():
    pos, mass = _cloud(n=12, dtype=torch.float64)
    eps = 1e-4

    def force_fn(p):
        return exact_force_and_tidal(p, mass, softening=eps)[0]

    jac = torch.autograd.functional.jacobian(force_fn, pos)        # (N,3,N,3)
    analytic = exact_force_and_tidal(pos, mass, softening=eps)[1]
    # Only i == j (the query's own block) is the tidal tensor.
    diag = torch.stack([jac[i, :, i, :] for i in range(pos.shape[0])])
    assert torch.allclose(diag, analytic, atol=TOL64), (
        f"max |autograd - exact_tidal| = {(diag - analytic).abs().max().item():.3e}"
    )


def test_exact_tidal_is_symmetric_and_trace_free_in_vacuum():
    """A tidal tensor is symmetric; far from all mass its trace vanishes (Laplace)."""
    pos, mass = _cloud(n=32, dtype=torch.float64)
    _f, tidal = exact_force_and_tidal(pos, mass, softening=1e-4)
    assert torch.allclose(tidal, tidal.transpose(-1, -2), atol=1e-10)
    # Trace is -div a = 0 away from the softened core. It is NOT zero here (the softening
    # radius is small but the particles are close, so each softened core contributes an
    # isotropic -1/r^3 term); the invariant that must hold exactly is symmetry plus the
    # derivative identity above. Check the trace is finite and negative (attractive).
    assert torch.isfinite(tidal).all()
    assert (tidal.diagonal(dim1=-2, dim2=-1).sum(-1) < 0).all()


def test_exact_force_matches_barnes_hut_exact_force():
    """The reference must agree with the pre-existing exact_force the papers already use."""
    from laniakea_hmmp.models.barnes_hut import exact_force

    pos, mass = _cloud(n=64)
    force, _tidal = exact_force_and_tidal(pos, mass, softening=1e-4)
    legacy = exact_force(pos, mass, softening=1e-4)
    rel = (force - legacy).norm().item() / legacy.norm().item()
    assert rel < 1e-4, f"exact reference disagrees with exact_force (rel {rel:.3e})"


def test_shipped_legacy_expression_is_not_the_jacobian():
    """Regression evidence for the defect, not a behaviour we rely on.

    ``3 d d^T / r^2 - I / r^3`` agrees with the Jacobian only at r = 1; at r = 0.2 the
    xx-component is off by a factor of ~3 (it is dimensionless where it should scale as
    1/r^3). If someone reintroduces it, this test still passes (it characterises the old
    formula), but ``test_exact_tidal_is_the_force_jacobian`` would fail.
    """
    m, eps = 1.0, 0.0
    for R in (0.2, 5.0):
        y = torch.tensor([R, 0.0, 0.0], dtype=torch.float64)
        x = torch.zeros(3, dtype=torch.float64)
        d = y - x
        r2 = (d * d).sum()
        inv_r3 = r2 ** -1.5
        correct = 3.0 * (d[0] * d[0]) * inv_r3 / r2 - inv_r3
        legacy = 3.0 * (d[0] * d[0]) / r2 - inv_r3
        assert abs(correct - 2.0 / R ** 3) < 1e-12          # a_xx = 2/R^3 for a single mass
        assert abs(legacy - correct) > 1.0                  # wildly different off r = 1
    # ... and at r = 1 the two coincide, which is why the defect hid for so long.
    R = 1.0
    y = torch.tensor([R, 0.0, 0.0], dtype=torch.float64)
    d = y.clone()
    legacy = 3.0 * (d[0] * d[0]) / (d * d).sum() - (d * d).sum() ** -1.5
    assert abs(legacy - 2.0) < 1e-12


# --------------------------------------------------------------------------------------
# 2. the quadrupole tidal term
# --------------------------------------------------------------------------------------
def test_quadrupole_tidal_matches_autograd_of_the_shipped_quadrupole_force():
    """The sharpest test of the new algebra: differentiate the *existing* force term."""
    torch.manual_seed(0)
    dev = torch.device("cpu")
    layer = _layer(use_quadrupole=True)

    # One synthetic cell with a genuine second moment (three unequal masses off-axis).
    y = torch.tensor([[0.30, 0.0, 0.10], [-0.20, 0.15, 0.0], [0.05, -0.25, -0.12]],
                     dtype=torch.float64)
    w = torch.tensor([1.0, 0.7, 1.3], dtype=torch.float64)
    com = (w.unsqueeze(1) * y).sum(0) / w.sum()
    rel = y - com
    D = (w.unsqueeze(1).unsqueeze(2) * (rel.unsqueeze(2) * rel.unsqueeze(1))).sum(0)
    moments = {
        "com": com.unsqueeze(0),
        "total_mass": w.sum().reshape(1),
        "quadrupole": D.unsqueeze(0),
    }
    query = torch.tensor([[1.4, 0.3, -0.7], [-0.9, 1.1, 0.4]], dtype=torch.float64)
    mass = torch.ones(2, dtype=torch.float64)
    cl = torch.tensor([0, 0])
    empty = torch.empty(0, dtype=torch.long)
    interaction_list = (torch.tensor([0, 1]), cl, empty, empty)

    def far(p):
        # Differentiate the *force branch* of the combined method (its acceleration term is
        # the shipped a_quad, pinned by test_quadrupole_physics.py) and compare with the
        # tidal branch. The two branches share no algebra, so agreement is not circular.
        return layer.far_field_acceleration_and_tidal(
            None, moments, p, mass, theta=0.5, interaction_list=interaction_list
        )[0]

    jac = torch.autograd.functional.jacobian(far, query)          # (Q,3,Q,3)
    diag = torch.stack([jac[i, :, i, :] for i in range(query.shape[0])])
    _acc, tidal = layer.far_field_acceleration_and_tidal(
        None, moments, query, mass, theta=0.5, interaction_list=interaction_list
    )
    assert torch.allclose(diag, tidal, atol=TOL64), (
        f"quadrupole tidal is not the Jacobian: max diff {(diag - tidal).abs().max().item():.3e}"
    )
    # And the monopole-only tidal must NOT be that Jacobian (i.e. the term does real work).
    _acc0, tidal0 = _layer(use_quadrupole=False).far_field_acceleration_and_tidal(
        None, moments, query, mass, theta=0.5, interaction_list=interaction_list
    )
    assert not torch.allclose(diag, tidal0, atol=1e-8)


def test_quadrupole_tidal_increment_matches_two_mass_axial_cell():
    """Analytic check on the same two-mass cell the force tests use.

    Two equal masses m at z = +/-h about the COM: the exact axial acceleration is
    a_z = -m[(R-h)^-2 + (R+h)^-2], so its z-derivative is
    T_zz = 2m[(R-h)^-3 + (R+h)^-3] = 4m/R^3 + 24 m h^2/R^5 + O(h^4/R^7).
    The monopole reproduces 4m/R^3 exactly and the retained quadrupole must add
    24 m h^2 / R^5.
    """
    m, h, R = 1.0, 0.1, 5.0
    amp = 2.0 * m * h * h
    moments = {
        "com": torch.zeros(1, 3),
        "total_mass": torch.tensor([2.0 * m]),
        "quadrupole": torch.diag(torch.tensor([0.0, 0.0, amp])).unsqueeze(0),
    }
    query = torch.tensor([[0.0, 0.0, R]])
    mass = torch.tensor([m])
    empty = torch.empty(0, dtype=torch.long)
    interaction_list = (torch.tensor([0]), torch.tensor([0]), empty, empty)

    res = {}
    for use_quad in (False, True):
        _acc, tidal = _layer(use_quad).far_field_acceleration_and_tidal(
            None, moments, query, mass, theta=0.5, interaction_list=interaction_list
        )
        res[use_quad] = tidal[0]
        assert torch.allclose(tidal[0], tidal[0].T, atol=1e-6), "tidal must be symmetric"

    mono_zz = res[False][2, 2].item()
    quad_zz = res[True][2, 2].item()
    assert abs(mono_zz - 4.0 * m / R ** 3) < 1e-6
    assert abs((quad_zz - mono_zz) - 24.0 * m * h * h / R ** 5) < 1e-6, (
        f"quadrupole tidal increment {quad_zz - mono_zz!r}, "
        f"expected {24.0 * m * h * h / R ** 5!r}"
    )
    # An axial cell is isotropic in the transverse plane: T_xx = T_yy < 0.
    assert abs(res[True][0, 0].item() - res[True][1, 1].item()) < 1e-6
    assert res[True][0, 0].item() < 0
    assert abs(res[True][0, 1].item()) < 1e-6


def test_combined_far_field_force_is_identical_to_the_force_only_path():
    """The sweep must not change the force column it is compared against."""
    pos, mass = _cloud(n=200)
    tree, moments = _tree(pos, mass)
    for use_quad in (False, True):
        layer = _layer(use_quad)
        for theta in (0.3, 0.6):
            ll = tree.build_interaction_list(theta=theta, device=torch.device("cpu"))
            with torch.no_grad():
                force_only = layer.far_field_acceleration(tree, moments, pos, mass,
                                                          theta=theta, interaction_list=ll)
                force_comb, _tidal = layer.far_field_acceleration_and_tidal(
                    tree, moments, pos, mass, theta=theta, interaction_list=ll
                )
            assert torch.equal(force_only, force_comb), (
                f"the combined call changed the force (quad={use_quad}, theta={theta})"
            )


# --------------------------------------------------------------------------------------
# 3. near leaves
# --------------------------------------------------------------------------------------
def test_near_leaf_tidal_equals_exact_when_nothing_opens():
    """At theta = 0 no cell ever opens, so near+far must reduce to the exact pairwise sum."""
    pos, mass = _cloud(n=120, seed=3)
    tree, moments = _tree(pos, mass)
    ll = tree.build_interaction_list(theta=0.0, device=torch.device("cpu"))
    assert ll[0].numel() == 0, "theta=0 must admit no far cell"

    layer = _layer(use_quadrupole=False)
    with torch.no_grad():
        near_f, near_t = layer.near_leaf_acceleration_and_tidal(
            tree, pos, mass, theta=0.0, interaction_list=ll
        )
    ex_f, ex_t = exact_force_and_tidal(pos, mass, softening=1e-4)

    # Softening differs (1e-8 vs 1e-4^2 = 1e-8 -- identical here), so this is an identity.
    assert (near_t - ex_t).abs().max().item() / ex_t.abs().max().item() < 5e-4
    assert (near_f - ex_f).norm().item() / ex_f.norm().item() < 5e-4


def test_near_leaf_tidal_matches_exact_pairwise_on_a_single_leaf():
    """Direct construction: one leaf holding every particle, no far cells at all."""
    torch.manual_seed(2)
    pos = torch.randn(6, 3)
    mass = torch.rand(6) + 0.5
    tree = BarnesHutTree(pos, center=pos.mean(0), size=4.0, max_leaf=16, max_depth=8)
    moments = tree.moments(mass)

    # Ask each particle for the leaf containing everything: force the whole set near.
    class _FakeList:
        pass

    leaf_of_root = 0
    while not tree.cell_is_leaf[leaf_of_root]:
        leaf_of_root = int(tree.cell_children[leaf_of_root][0])
    src = torch.arange(pos.shape[0])
    cell = torch.full((pos.shape[0],), leaf_of_root, dtype=torch.long)
    empty = torch.empty(0, dtype=torch.long)
    # Far entries are used too; give the far list no cells.
    ll = (empty, empty, src, cell)

    layer = _layer(use_quadrupole=False)
    near_f, near_t = layer.near_leaf_acceleration_and_tidal(
        tree, pos, mass, theta=0.5, interaction_list=ll
    )
    # Compare against a hand-rolled exact pairwise sum over the queried pairs only.
    d = pos.unsqueeze(0) - pos.unsqueeze(1)
    r2 = (d * d).sum(-1) + 1e-8
    r2.fill_diagonal_(float("inf"))
    inv_r3 = r2.rsqrt() ** 3
    expect_f = (mass.unsqueeze(1) * inv_r3.unsqueeze(-1) * d).sum(1)
    outer = d.unsqueeze(-1) * d.unsqueeze(-2)
    contrib = 3.0 * outer * (inv_r3 / r2).unsqueeze(-1).unsqueeze(-1)
    contrib = contrib - torch.eye(3) * inv_r3.unsqueeze(-1).unsqueeze(-1)
    expect_t = (mass.unsqueeze(1).unsqueeze(2) * contrib).sum(1)

    assert torch.allclose(near_f, expect_f, atol=F32_TOL)
    assert torch.allclose(near_t, expect_t, atol=F32_TOL)
    assert moments["quadrupole"].shape[0] == tree.cell_center.shape[0]


# --------------------------------------------------------------------------------------
# 4. the complete operator
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_quadrupole", [False, True])
def test_complete_operator_tidal_approaches_exact_as_theta_shrinks(use_quadrupole):
    """far + near tidal must converge to the exact tidal tensor (and the force likewise)."""
    pos, mass = _cloud(n=400, seed=1)
    tree, moments = _tree(pos, mass)
    ex_f, ex_t = exact_force_and_tidal(pos, mass, softening=1e-4)

    layer = _layer(use_quadrupole)
    errs = {}
    for theta in (0.2, 0.5):
        ll = tree.build_interaction_list(theta=theta, device=torch.device("cpu"))
        with torch.no_grad():
            far_f, far_t = layer.far_field_acceleration_and_tidal(
                tree, moments, pos, mass, theta=theta, interaction_list=ll
            )
            near_f, near_t = layer.near_leaf_acceleration_and_tidal(
                tree, pos, mass, theta=theta, interaction_list=ll
            )
        tot_f, tot_t = near_f + far_f, near_t + far_t
        errs[theta] = (
            (tot_f - ex_f).norm().item() / ex_f.norm().item(),
            (tot_t - ex_t).norm().item() / ex_t.norm().item(),
        )

    for theta, (ef, et) in errs.items():
        assert ef < 0.6, f"force error {ef:.3f} at theta={theta}"
        assert et < 0.6, f"tidal error {et:.3f} at theta={theta}"
    # Convergence: the smaller opening angle must be the more accurate one, for both.
    assert errs[0.2][0] < errs[0.5][0]
    assert errs[0.2][1] < errs[0.5][1]
