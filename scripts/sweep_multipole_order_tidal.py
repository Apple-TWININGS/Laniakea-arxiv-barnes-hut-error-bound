#!/usr/bin/env python
"""§5: the multipole-ORDER sweep, extended to the tidal tensor (second derivative).

The blocking question
---------------------
`scripts/sweep_multipole_order.py` measured, for the **force**, that

    ratio(theta) := mean_rel_err(p=0) / mean_rel_err(p=1)   satisfies   ratio * theta ~ const,

which is the signature of the omitted multipole term being exactly one order above the
retained one (the truncation ratio between consecutive orders scales as the cell-size-to-
distance ratio, i.e. as 1/theta). If the same holds for the **tidal tensor**, route B is
viable: the second-derivative remainder inherits the same order shift, and a bound can be
built by differentiating the classical force remainder. If it does not hold, that is itself
the result (same tree, same interaction list, same query nodes -- only the derivative count
differs).

What this script does
---------------------
For one synthetic cloud, one tree and one interaction list per theta, it computes the
complete operator (exact near-leaf direct sums + truncated far-field multipole) in **both**
derivative counts:

    p = 0   monopole only            (use_quadrupole=False)
    p = 1   monopole + quadrupole    (use_quadrupole=True)

and scores both force and tidal tensor against the exact N-body reference
(`laniakea_hmmp.models.exact_field`), which is itself pinned to the autograd Jacobian of the
exact force by `tests/test_tidal_tensor.py`.

Two ratios are reported, deliberately
-------------------------------------
1. ``ratio_rel_*`` -- the per-node **relative** error ratio, directly comparable with the
   existing force artifact. NOTE: the denominator ``||T_total||`` is itself theta-dependent
   (as theta grows a different set of particles is summed exactly in the near field), which
   is exactly the confound flagged in HANDOFF §2 (R1). It cancels *within* the ratio (both
   orders share the same denominator per node), but the normalised errors are not comparable
   *across* theta.
2. ``ratio_abs_*`` -- the **absolute** truncation-error ratio, ``||operator - exact||`` over
   the whole query set with no normalisation at all. The near-field part is an exact direct
   sum in both cases and cancels identically, so this is a pure far-field remainder norm: the
   theta-dependent denominator cannot enter it at all. This is the clean version of the test.

Usage:
    python scripts/sweep_multipole_order_tidal.py --n-list 2000 8000 32000 \
        --theta-list 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 \
        --out outputs/multipole_order_sweep_tidal.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from laniakea_hmmp.data.synthetic import generate_particle_cloud
from laniakea_hmmp.models.barnes_hut import BarnesHutTree
from laniakea_hmmp.models.exact_field import exact_force_and_tidal
from laniakea_hmmp.models.hmmp import HMMP


def operator_force_and_tidal(layer, tree, moments, pos, mass, theta, interaction_list):
    """Complete HMMP operator: exact near-leaf direct sums + far-field multipole.

    Returns ``(force, tidal)``; the tidal tensor is ``T[i, a, b] = d a[i, a] / d x[i, b]``,
    the same object (and the same convention) as ``exact_field.exact_force_and_tidal``.
    """
    far_f, far_t = layer.far_field_acceleration_and_tidal(
        tree, moments, pos, mass, theta=theta, interaction_list=interaction_list
    )
    near_f, near_t = layer.near_leaf_acceleration_and_tidal(
        tree, pos, mass, theta=theta, interaction_list=interaction_list
    )
    return near_f + far_f, near_t + far_t


def _stats(approx: torch.Tensor, exact: torch.Tensor, denom_exact: torch.Tensor) -> dict:
    """Relative-error distribution, absolute error norm, and direction agreement."""
    diff = approx - exact
    flat_a = approx.flatten(1) if approx.dim() > 2 else approx
    flat_e = exact.flatten(1) if exact.dim() > 2 else exact
    num = diff.flatten(1).norm(dim=-1) if diff.dim() > 2 else diff.norm(dim=-1)
    den = flat_e.norm(dim=-1).clamp_min(1e-30)
    per = (num / den).detach().cpu()
    return {
        "rel_err_mean": float(per.mean().item()),
        "rel_err_median": float(per.median().item()),
        "rel_err_p90": float(torch.quantile(per, 0.90).item()),
        "rel_err_max": float(per.max().item()),
        "global_rel_err": float(num.norm().item() / den.norm().clamp_min(1e-30).item()),
        "abs_err_norm": float(diff.norm().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(
            flat_a.flatten(), flat_e.flatten(), dim=0).item()),
        "exact_norm": float(flatten_norm(exact)),
        "exact_field_norm_for_norm_ratio": float(flatten_norm(denom_exact)),
    }


def flatten_norm(t: torch.Tensor) -> float:
    return float(t.detach().norm().item())


def _loglog_slope(thetas: list[float], values: list[float]) -> float | None:
    """Slope of log(value) vs log(theta): the empirical power of the truncation error."""
    pairs = [(t, v) for t, v in zip(thetas, values) if t > 0 and v is not None and v > 0]
    if len(pairs) < 2:
        return None
    x = np.log(np.array([p[0] for p in pairs], dtype=np.float64))
    y = np.log(np.array([p[1] for p in pairs], dtype=np.float64))
    return float(np.polyfit(x, y, 1)[0])


def _summarise(rows: list[dict], prefix: str) -> dict:
    """ratio * theta per theta, and its scatter -- for one quantity (force or tidal)."""
    per_theta: dict[float, list[float]] = {}
    per_theta_rel: dict[float, list[float]] = {}
    for r in rows:
        if r.get(f"ratio_abs_{prefix}") is None:
            continue
        per_theta.setdefault(r["theta"], []).append(r[f"ratio_abs_{prefix}"])
        if r.get(f"ratio_rel_{prefix}") is not None:
            per_theta_rel.setdefault(r["theta"], []).append(r[f"ratio_rel_{prefix}"])

    out: dict = {
        f"ratio_abs_times_theta_by_theta_{prefix}": {},
        f"ratio_rel_times_theta_by_theta_{prefix}": {},
    }
    for label, src, key in (
        ("abs", per_theta, f"ratio_abs_times_theta_by_theta_{prefix}"),
        ("rel", per_theta_rel, f"ratio_rel_times_theta_by_theta_{prefix}"),
    ):
        out[key] = {f"{t:.4f}": float(np.mean(v) * t) for t, v in sorted(src.items())}
        vals = np.array(sorted(out[key].values()))
        if vals.size:
            out[f"ratio_{label}_times_theta_mean_{prefix}"] = float(vals.mean())
            out[f"ratio_{label}_times_theta_std_{prefix}"] = (
                float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            )
            out[f"ratio_{label}_times_theta_rel_scatter_{prefix}"] = (
                float(vals.std(ddof=1) / vals.mean()) if vals.size > 1 else 0.0
            )
    thetas = sorted({r["theta"] for r in rows})
    out[f"theta_power_abs_{prefix}"] = _loglog_slope(
        thetas, [next((r[f"p0_{prefix}_abs_err_norm"] for r in rows if r["theta"] == t), None)
                 for t in thetas])
    out[f"theta_power_rel_{prefix}"] = _loglog_slope(
        thetas, [next((r[f"p0_{prefix}_rel_err_mean"] for r in rows if r["theta"] == t), None)
                 for t in thetas])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[2000, 8000, 32000])
    ap.add_argument("--grid-n", type=int, default=32)
    ap.add_argument("--theta-list", type=float, nargs="+",
                    default=[0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--softening", type=float, default=1e-4)
    ap.add_argument("--max-leaf", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--far-batch", type=int, default=1 << 20,
                    help="interaction entries per batch (matches the force-only sweep default)")
    ap.add_argument("--slope-max-theta", type=float, default=0.5,
                    help="theta ceiling for the log-log power fit (classical regime)")
    ap.add_argument("--opening-mode", default="com", choices=["com", "center", "surface"],
                    help="acceptance test; 'com' (distance to the cell centre of mass) is the "
                         "published criterion. See docs/PROVENANCE.md, section 3.5.")
    ap.add_argument("--legacy-geometry", action="store_true",
                    help="reproduce the pre-D8 tree construction")
    ap.add_argument("--out", default="outputs/multipole_order_sweep_tidal.json")
    args = ap.parse_args()

    dev = torch.device("cpu")  # deterministic; N is small
    torch.manual_seed(args.seed)

    out = {
        "seed": args.seed,
        "softening": args.softening,
        "grid_n": args.grid_n,
        "n_list": args.n_list,
        "theta_list": args.theta_list,
        "max_leaf": args.max_leaf,
        "max_depth": args.max_depth,
        "far_batch": args.far_batch,
        "tidal_convention": ("T[i,a,b] = d a[i,a] / d x[i,b], from exact_field.exact_force_and_tidal "
                             "(the Jacobian of the force; see docs/PROVENANCE.md 3.4)"),
        "rows": [],
    }

    t_start = time.time()
    for n in args.n_list:
        p = generate_particle_cloud(n_particles=n, grid_n=args.grid_n, seed=args.seed)
        pos = torch.from_numpy(p["pos"]).float().to(dev)
        mass = torch.from_numpy(p["mass"]).float().to(dev)
        center = pos.mean(0).detach()
        size = float((pos.max(0).values - pos.min(0).values).max().item()) * 1.05

        t0 = time.time()
        exact_f, exact_t = exact_force_and_tidal(pos, mass, softening=args.softening)
        t_exact = time.time() - t0

        # Bounding-box root and octant centres (defects D8a/D8b fixed); ``size`` is a lower bound.
        tree = BarnesHutTree(pos, size=size, max_leaf=args.max_leaf,
                             max_depth=args.max_depth,
                             legacy_geometry=getattr(args, "legacy_geometry", False))
        moments = tree.moments(mass)
        has_quad = bool("quadrupole" in moments and moments["quadrupole"] is not None)

        layers = {
            0: HMMP(node_dim=2, hidden_dim=8, n_layers=1, max_leaf=args.max_leaf,
                    max_depth=args.max_depth, use_quadrupole=False,
                    far_batch=args.far_batch).to(dev).layers[0],
            1: HMMP(node_dim=2, hidden_dim=8, n_layers=1, max_leaf=args.max_leaf,
                    max_depth=args.max_depth, use_quadrupole=True,
                    far_batch=args.far_batch).to(dev).layers[0],
        }

        # Sanity: the exact reference must be physically plausible before anything is read
        # from it (HANDOFF 10.3: two earlier probes returned confident nonsense).
        print(f"n={n}: |a_exact|={flatten_norm(exact_f):.6e}  |T_exact|={flatten_norm(exact_t):.6e}  "
              f"mass_std={float(mass.std().item()):.3e}  (exact ref {t_exact:.1f}s)")

        for theta in args.theta_list:
            t0 = time.time()
            ll = tree.build_interaction_list(theta=theta, device=dev,
                                             opening_mode=args.opening_mode, com=moments["com"])
            row = {"n": n, "theta": theta, "has_quadrupole_moment": has_quad,
                   "n_far_entries": int(ll[0].numel()), "n_near_entries": int(ll[2].numel())}

            res = {}
            for order in (0, 1):
                tot_f, tot_t = operator_force_and_tidal(
                    layers[order], tree, moments, pos, mass, theta, ll
                )
                res[order] = {
                    "force": _stats(tot_f, exact_f, exact_f),
                    "tidal": _stats(tot_t, exact_t, exact_t),
                }
                if not (torch.isfinite(tot_f).all() and torch.isfinite(tot_t).all()):
                    raise RuntimeError(f"non-finite operator output at n={n}, theta={theta}, p={order}")

            for quantity, prefix in (("force", "force"), ("tidal", "tidal")):
                for field, key in (("rel_err_mean", "rel_err_mean"),
                                   ("rel_err_median", "rel_err_median"),
                                   ("global_rel_err", "global_rel_err"),
                                   ("abs_err_norm", "abs_err_norm"),
                                   ("cosine", "cosine")):
                    row[f"p0_{prefix}_{key}"] = res[0][quantity][field]
                    row[f"p1_{prefix}_{key}"] = res[1][quantity][field]

            for prefix in ("force", "tidal"):
                p0_rel, p1_rel = res[0][prefix]["rel_err_mean"], res[1][prefix]["rel_err_mean"]
                p0_abs, p1_abs = res[0][prefix]["abs_err_norm"], res[1][prefix]["abs_err_norm"]
                row[f"ratio_rel_{prefix}"] = (p0_rel / p1_rel) if p1_rel > 0 else None
                row[f"ratio_abs_{prefix}"] = (p0_abs / p1_abs) if p1_abs > 0 else None
                row[f"ratio_rel_times_theta_{prefix}"] = (
                    row[f"ratio_rel_{prefix}"] * theta if row[f"ratio_rel_{prefix}"] else None)
                row[f"ratio_abs_times_theta_{prefix}"] = (
                    row[f"ratio_abs_{prefix}"] * theta if row[f"ratio_abs_{prefix}"] else None)

            row["exact_force_norm"] = res[0]["force"]["exact_norm"]
            row["exact_tidal_norm"] = res[0]["tidal"]["exact_norm"]
            row["time_s"] = time.time() - t0
            out["rows"].append(row)

            print(f"  theta={theta:.2f}  force p0={row['p0_force_rel_err_mean']:.6f} "
                  f"p1={row['p1_force_rel_err_mean']:.6f}  r*th={row['ratio_rel_times_theta_force']:.4f} | "
                  f"tidal p0={row['p0_tidal_rel_err_mean']:.6f} p1={row['p1_tidal_rel_err_mean']:.6f}  "
                  f"r*th={row['ratio_rel_times_theta_tidal']:.4f} | "
                  f"abs r*th: f={row['ratio_abs_times_theta_force']:.4f} "
                  f"t={row['ratio_abs_times_theta_tidal']:.4f}")

    # ---- validity filter (same convention as the force sweep) ---------------------------
    valid = [r for r in out["rows"]
             if r["p0_force_rel_err_mean"] < 1.0 and r["p1_force_rel_err_mean"] < 1.0]
    broken = [r for r in out["rows"] if r not in valid]
    out["n_valid_cells"] = len(valid)
    out["n_broken_cells"] = len(broken)
    out["broken_cells"] = [{"n": r["n"], "theta": r["theta"],
                            "p0_force_rel_err_mean": r["p0_force_rel_err_mean"],
                            "p1_force_rel_err_mean": r["p1_force_rel_err_mean"]} for r in broken]
    # Tidal-specific breakdown flag: the second derivative can blow up where the force is fine.
    out["tidal_broken_cells"] = [
        {"n": r["n"], "theta": r["theta"], "p0_tidal_rel_err_mean": r["p0_tidal_rel_err_mean"],
         "p1_tidal_rel_err_mean": r["p1_tidal_rel_err_mean"]}
        for r in out["rows"]
        if max(r["p0_tidal_rel_err_mean"], r["p1_tidal_rel_err_mean"]) >= 1.0
    ]

    slope_rows = [r for r in valid if r["theta"] <= args.slope_max_theta]
    out.update(_summarise(slope_rows, "force"))
    out.update(_summarise(slope_rows, "tidal"))
    out["slope_max_theta"] = args.slope_max_theta
    out["time_total_s"] = time.time() - t_start

    print("\n--- summary (valid cells with theta <= %.2f) ---" % args.slope_max_theta)
    for prefix in ("force", "tidal"):
        print(f"{prefix}:")
        for label in ("abs", "rel"):
            key = f"ratio_{label}_times_theta_by_theta_{prefix}"
            for t, v in out[key].items():
                print(f"   theta={t}  ratio_{label}*theta={v:.6f}")
            mean = out.get(f"ratio_{label}_times_theta_mean_{prefix}")
            scatter = out.get(f"ratio_{label}_times_theta_rel_scatter_{prefix}")
            if mean is not None:
                print(f"   ratio_{label}*theta mean={mean:.6f}  rel scatter={scatter:.4%}")
        print(f"   empirical theta power: abs={out[f'theta_power_abs_{prefix}']}  "
              f"rel={out[f'theta_power_rel_{prefix}']}   "
              f"(classical force prediction 3.0)")
    if broken:
        print("breakdown cells excluded: "
              + ", ".join(f"n={r['n']},theta={r['theta']}" for r in broken))
    if out["tidal_broken_cells"]:
        print(f"cells where the TIDAL error exceeds 100%: {len(out['tidal_broken_cells'])}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}  ({out['time_total_s']:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
