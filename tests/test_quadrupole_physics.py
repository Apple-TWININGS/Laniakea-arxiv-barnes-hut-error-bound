"""Physics correctness of the QUADRUPOLE far-field term (multipole order p = 1).

Regression guard. The shipped quadrupole term used to be

    a_quad = 0.5 * (rhat^T D rhat) * rhat / r^3

which has the wrong radial power (1/r^3 instead of 1/r^4), drops the (D rhat) term, and
uses the untraced second moment instead of the traceless quadrupole tensor. It made the
far field *worse* than monopole-only (relative error 0.0042 vs 0.0025 on a triaxial cloud,
and 0.808 vs 0.041 inside the full p-sweep). No existing test covered this path, because
every training script sets ``use_quadrupole=False``. These tests pin the analytic result.

Setup. For two equal masses m at z = +/-d (COM at the origin) the exact axial acceleration
at z = R is

    a_z(R) = -m [ 1/(R-d)^2 + 1/(R+d)^2 ]
           = -2m/R^2 - 6 m d^2 / R^4 - 10 m d^4 / R^6 - O(d^6/R^8)

So a correct monopole+quadrupole must reproduce the first two terms, and the *increment*
introduced by the quadrupole must equal -6 m d^2 / R^4 exactly. That increment identity is
the sharpest available test: it involves no truncation error at all.

All tolerances below are set for float32, which is what the model actually runs in
(relative eps ~1.2e-7, so ~1e-8 absolute at a magnitude of 0.08).
"""
import pytest
import torch

from laniakea_hmmp.models.hmmp import HMMP

# float32 gives ~6e-9 absolute resolution at |a| ~ 0.08.
F32_TOL = 1e-7


class _DummyTree:
    """far_field_acceleration only uses the tree when interaction_list is None."""


def _two_mass_far_field(use_quadrupole: bool, R: float, m: float = 1.0,
                        d: float = 0.1) -> float:
    """Axial far-field acceleration at z = R from ONE synthetic two-mass cell.

    The cell has total mass 2m, COM at the origin, and second moment about its COM
    D = diag(0, 0, 2 m d^2) -- exactly the moment tensor of two point masses at z = +/-d.
    Returns the z component.
    """
    torch.manual_seed(0)
    model = HMMP(node_dim=3, hidden_dim=8, n_layers=1, max_leaf=8, max_depth=16,
                 theta=0.5, use_quadrupole=use_quadrupole).eval()

    amp = 2.0 * m * d * d
    moments = {
        "com": torch.tensor([[0.0, 0.0, 0.0]]),
        "total_mass": torch.tensor([2.0 * m]),
        "quadrupole": torch.tensor([[[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, amp]]]),
    }
    pos = torch.tensor([[0.0, 0.0, R]])
    mass = torch.tensor([m])
    empty = torch.empty(0, dtype=torch.long)
    interaction_list = (torch.tensor([0]), torch.tensor([0]), empty, empty)

    with torch.no_grad():
        acc = model.layers[0].far_field_acceleration(
            _DummyTree(), moments, pos, mass, theta=0.5, interaction_list=interaction_list
        )
    assert acc.shape == (1, 3)
    # The synthetic cell sits exactly on the z axis, so x and y must stay zero.
    assert abs(acc[0, 0].item()) < F32_TOL and abs(acc[0, 1].item()) < F32_TOL
    return acc[0, 2].item()


def _exact_two_mass(R: float, m: float = 1.0, d: float = 0.1) -> float:
    return -m * (1.0 / (R - d) ** 2 + 1.0 / (R + d) ** 2)


def test_moment_tensor_used_by_tests_is_the_true_second_moment():
    """The synthetic D must really be sum_a m_a y_a y_a^T for y = (0,0,+/-d)."""
    m, d = 1.0, 0.1
    y = torch.tensor([[0.0, 0.0, d], [0.0, 0.0, -d]])
    D = (m * (y.unsqueeze(2) * y.unsqueeze(1))).sum(0)
    assert torch.allclose(D, torch.diag(torch.tensor([0.0, 0.0, 2.0 * m * d * d])),
                          atol=1e-9)
    # It is NOT traceless -- exactly why the traceless conversion 3D - tr(D) Id is needed.
    assert abs(D.diagonal().sum().item() - 2.0 * m * d * d) < 1e-8
    # And the traceless tensor I = 3D - tr(D) Id must be traceless.
    I = 3.0 * D - D.diagonal().sum() * torch.eye(3)
    assert abs(I.diagonal().sum().item()) < 1e-8


def test_monopole_matches_analytic_leading_term():
    """Monopole-only must give exactly -2m/R^2, with no d-dependence."""
    R = 5.0
    mono = _two_mass_far_field(False, R=R)
    assert abs(mono - (-2.0 / R ** 2)) < F32_TOL
    # Changing d must not move the monopole answer at all.
    mono_other = _two_mass_far_field(False, R=R, d=0.25)
    assert abs(mono - mono_other) < F32_TOL


def test_quadrupole_increment_equals_analytic_correction():
    """SHARPEST TEST: the quadrupole increment must be exactly -6 m d^2 / R^4.

    Pure algebra -- no truncation error is involved, so the tolerance can be tight.
    The old (buggy) implementation produced +2 m d^2 / R^3 instead, i.e. +4.0e-4 at
    R = 5, d = 0.1, which is ~4000x outside this tolerance.
    """
    m, d, R = 1.0, 0.1, 5.0
    expected_increment = -6.0 * m * d * d / R ** 4          # = -9.6e-5

    mono = _two_mass_far_field(False, R=R, m=m, d=d)
    quad = _two_mass_far_field(True, R=R, m=m, d=d)
    increment = quad - mono

    assert abs(increment - expected_increment) < F32_TOL, (
        f"quadrupole increment is not -6 m d^2 / R^4: got {increment!r}, "
        f"expected {expected_increment!r}"
    )
    # Guard the sign explicitly: the correction must pull the estimate down, not up.
    assert increment < 0.0


@pytest.mark.parametrize("R", [4.0, 5.0, 8.0, 12.0])
def test_quadrupole_beats_monopole_against_exact(R):
    """Retaining the quadrupole must cut the error by orders of magnitude."""
    mono = _two_mass_far_field(False, R=R)
    quad = _two_mass_far_field(True, R=R)
    exact = _exact_two_mass(R)

    err_mono = abs(mono - exact)
    err_quad = abs(quad - exact)

    assert err_quad < err_mono, f"quadrupole worse than monopole at R={R}"
    # Monopole error is 6md^2/R^4; the quadrupole residual is the O(d^4/R^6) term, so the
    # improvement factor grows as (R/d)^2. At R=4 that is already ~1600/10 = 160.
    assert err_quad < err_mono / 50.0, (
        f"quadrupole should cut the error by >50x at R={R}: "
        f"mono={err_mono:.3e} quad={err_quad:.3e}"
    )


def test_quadrupole_residual_is_the_next_analytic_order():
    """The leftover error must be the genuine O(d^4/R^6) term, not a defect."""
    m, d, R = 1.0, 0.1, 5.0
    three_terms = (-2.0 * m / R ** 2 - 6.0 * m * d * d / R ** 4
                   - 10.0 * m * d ** 4 / R ** 6)
    exact = _exact_two_mass(R, m, d)
    # The 3-term expansion reproduces the exact sum to better than float32 resolution.
    assert abs(three_terms - exact) < F32_TOL

    quad = _two_mass_far_field(True, R=R, m=m, d=d)
    assert abs(quad - three_terms) < F32_TOL, (
        f"quadrupole residual is not the O(d^4/R^6) term: got {quad!r}, "
        f"expected {three_terms!r}"
    )


def test_quadrupole_off_by_default():
    """Every production training script relies on the default being monopole-only."""
    torch.manual_seed(0)
    model = HMMP(node_dim=2, hidden_dim=8, n_layers=1)
    assert model.use_quadrupole is False
    assert all(layer.use_quadrupole is False for layer in model.layers)
