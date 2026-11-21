"""Synthetic cosmological density-field generator.

This produces a physically-motivated, reproducible cosmic density field *without*
needing hundreds of GB of N-body data, so the full HMMP pipeline can be exercised
end-to-end on a single workstation for smoke tests, and on 2x A800 for larger runs
before swapping in real dark-matter snapshots.

Approach
--------
1. Realise a Gaussian random field on a periodic 3D grid with a
   LCDM-like linear power spectrum (a CDM ``T(k)^2`` transfer function with a
   simple Eisenstein-Hu-style fit is used; see :func:`_transfer_function`).
2. Convert the field to an overdensity ``delta = rho/rho_bar - 1``.
3. Sample particle positions with probability proportional to the local density,
   producing clustered particles (nodes and filaments) rather than uniform points.
4. Return the *true* smoothed density at each particle, which is the target HMMP
   learns to reconstruct from graph structure alone.

Everything is deterministic given a ``seed``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["generate_density_field", "generate_particle_cloud"]

G = 6.67430e-11  # m^3 kg^-1 s^-2, gravitational constant


def _transfer_function(k: np.ndarray, sigma_8: float = 0.81) -> np.ndarray:
    """Eisenstein-Hu-like CDM transfer function (normalised to sigma_8 at a pivot).

    We use a widely-used fitting form so that the power spectrum has the correct
    broad-band shape: power increasing on large scales, turnover near the
    matter-radiation equality scale, and a ~k^-3 high-k falloff.
    """
    # Box wavenumber is in units of 1/L; rescale to a physical-ish axis.
    theta = k / 0.02
    logq = np.log(2.0 * np.exp(1.0) * theta)
    # Minimal fitting polynomial for the CDM transfer function (Bardeen et al. style).
    t = (
        logq
        / (
            logq
            + 0.5 * np.exp(
                1.43 * logq + 0.338 * logq**2 + 0.112 * logq**3
            )
            + 0.87 * logq**2
            + 1.0
        )
    )
    t = np.clip(t, 0.0, None)
    # Normalise so that the resulting RMS is order unity at the pivot scale.
    return t * sigma_8


def _power_spectrum(k: np.ndarray) -> np.ndarray:
    """Linearly-scaled matter power spectrum ``P(k) = A k^n T(k)^2``."""
    n = 1.0  # primordial tilt
    return k**n * _transfer_function(k) ** 2


def generate_density_field(
    grid_n: int = 64,
    box_len: float = 100.0,
    seed: int = 0,
    sigma_8: float = 0.81,
) -> dict[str, np.ndarray]:
    """Generate a periodic Gaussian random density field on a ``grid_n^3`` grid.

    Returns a dict with ``delta`` (overdensity field), ``grid_n``, ``box_len`` and
    the grid ``x/y/z`` coordinate arrays.
    """
    rng = np.random.default_rng(seed)

    # Nyquist-limited dimensionless wavenumbers (positive frequencies only).
    kf = 2.0 * np.pi / box_len
    freqs = np.fft.fftfreq(grid_n, d=box_len / grid_n)  # wrapped
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    k = np.sqrt(kx**2 + ky**2 + kz**2)

    # Set zero mode to zero frequency exactly (avoid divide-by-zero).
    k0 = np.where(k > 0, k, 1.0)
    pk = _power_spectrum(k0)
    pk = np.where(k > 0, pk, 0.0)

    # Gaussian random realisation: real and imaginary parts are N(0, P(k)/2).
    sqrt_pk = np.sqrt(pk)
    real = rng.standard_normal(k.shape) * sqrt_pk
    imag = rng.standard_normal(k.shape) * sqrt_pk
    fk = (real + 1j * imag) / np.sqrt(2.0)
    fk[0, 0, 0] = 0.0  # remove the background (mean)

    # Inverse FFT -> real-space Gaussian field.
    field = np.real(np.fft.ifftn(fk)) * (grid_n**3)

    # Standardise to zero mean, unit variance, then map to overdensity.
    field = (field - field.mean()) / (field.std() + 1e-12)
    delta = np.exp(0.6 * field) - 1.0  # lognormal-ish -> strictly positive density
    delta /= delta.std() + 1e-12

    coords = np.linspace(0.0, box_len, grid_n, endpoint=False)
    return {
        "delta": delta,
        "grid_n": grid_n,
        "box_len": box_len,
        "grid_coords": coords,
    }


def generate_particle_cloud(
    n_particles: int = 5000,
    grid_n: int = 64,
    box_len: float = 100.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Sample clustered particles from a synthetic density field.

    Returns dict with ``pos`` (N,3), ``mass`` (N,), ``velocity`` (N,3),
    ``true_density`` (N,) and the underlying ``delta`` field.
    """
    field = generate_density_field(grid_n=grid_n, box_len=box_len, seed=seed)
    delta = field["delta"]
    grid_n = field["grid_n"]

    # Density at each cell is proportional to (1 + delta), floor at a small value.
    density = np.clip(1.0 + delta, 1e-3, None)
    flat_density = density.ravel()
    prob = flat_density / flat_density.sum()

    # Importance-sample particle indices.
    idx = np.random.default_rng(seed).choice(
        flat_density.size, size=n_particles, replace=True, p=prob
    )
    i, j, k = np.unravel_index(idx, density.shape)

    # Jitter within each cell for continuous positions, wrap periodically.
    rng = np.random.default_rng(seed + 1)
    cell = box_len / grid_n
    jitter = rng.uniform(0.0, cell, size=(n_particles, 3))
    base = (np.stack([i, j, k], axis=1) + 0.0) * cell
    pos = (base + jitter) % box_len

    # True (smoothed) density target at each particle, from the grid.
    true_density = density[i, j, k]

    # Mass: proportional to local density times a unit particle mass.
    mass = true_density / true_density.mean()

    # Bulk-flow velocity estimate from density gradient (simple continuity argument).
    grad = np.gradient(delta, box_len / grid_n)
    vel = np.stack([g[i, j, k] for g in grad], axis=1)
    vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12)
    velocity = rng.uniform(-0.5, 0.5, size=(n_particles, 1)) * vel

    return {
        "pos": pos.astype(np.float32),
        "mass": mass.astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "true_density": true_density.astype(np.float32),
        "delta": delta,
        "box_len": box_len,
    }
