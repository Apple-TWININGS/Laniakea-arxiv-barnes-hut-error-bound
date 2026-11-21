#!/usr/bin/env python
"""A real multipole-ORDER sweep (p = 0 vs p = 1), not an opening-angle sweep.

Why this exists: `scripts/scaling.py`'s "error vs order" curve actually varies the
*opening angle* theta while holding the multipole truncation fixed at monopole-only
(`use_quadrupole=False`). So it measures the theta-dependence of the error, not the
dependence on the multipole order p, and the paper has to say so. This script closes that
gap directly: it evaluates the complete HMMP force (near-leaf exact + far-field multipole)
against a brute-force exact N-body force, once with the quadrupole term retained and once
without, on the same particles and the same tree.

    p = 0  ->  monopole only         (use_quadrupole=False)
    p = 1  ->  monopole + quadrupole (use_quadrupole=True)

Proposition 3.1 bounds the far-field truncation error of order p by
gamma * C_p * (R/r)^(p+2) * ||h|| * (1 - alpha^K) / (1 - alpha). The ratio between
consecutive orders therefore scales as R/r, and because Barnes-Hut only treats a cell as
"far" when ``cell_size / r < theta``, the measured ratio p=0/p=1 should be proportional to
1/theta. That is a falsifiable prediction, so the script reports ``ratio_times_theta`` for
every (n, theta) cell and checks whether it is roughly constant in theta.

Usage:
    python scripts/sweep_multipole_order.py --n-list 4000 16000 \
        --theta-list 0.3 0.5 0.7 1.0 --out outputs/multipole_order_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from laniakea_hmmp.data.synthetic import generate_particle_cloud
from laniakea_hmmp.models.barnes_hut import BarnesHutTree
from laniakea_hmmp.models.hmmp import HMMP


def exact_force(pos: torch.Tensor, mass: torch.Tensor, softening: float = 1e-4,
                chunk: int = 2048) -> torch.Tensor:
    """Brute-force O(N^2) Newtonian acceleration, evaluated in chunks.

    Chunking keeps the (chunk, N, 3) intermediate bounded, so this stays usable at
    N = 32000 (an un-chunked (N, N, 3) float32 tensor would need ~12 GB).
    """
    n = pos.shape[0]
    out = torch.empty(n, 3, dtype=pos.dtype, device=pos.device)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = pos.unsqueeze(0) - pos[a:b].unsqueeze(1)        # (b-a, N, 3)
        r2 = (d ** 2).sum(-1) + softening ** 2
        inv_r3 = r2.rsqrt() / r2                            # 1/r^3
        w = (mass.unsqueeze(0) * inv_r3).unsqueeze(-1)
        out[a:b] = (w * d).sum(1)
        del d, r2, inv_r3, w
    return out


def hmmp_force(model, tree, moments, pos, mass, theta, device,
               opening_mode="center") -> torch.Tensor:
    layer = model.layers[0]
    ll = tree.build_interaction_list(theta=theta, device=device, opening_mode=opening_mode,
                                     com=moments["com"])
    with torch.no_grad():
        far = layer.far_field_acceleration(tree, moments, pos, mass, theta=theta,
                                           interaction_list=ll)
        leaf = layer.near_leaf_acceleration(tree, pos, mass, theta=theta,
                                            interaction_list=ll)
    return leaf + far


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000,
                    help="single particle count (kept for backwards compatibility)")
    ap.add_argument("--n-list", type=int, nargs="+", default=None)
    ap.add_argument("--grid-n", type=int, default=32)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--theta-list", type=float, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--softening", type=float, default=1e-4)
    ap.add_argument("--opening-mode", default="com", choices=["com", "center", "surface"],
                    help="acceptance test; 'com' is the published criterion and the default "
                         "here. See docs/PROVENANCE.md, section 3.5.")
    ap.add_argument("--legacy-geometry", action="store_true",
                    help="reproduce the pre-D8 tree construction")
    ap.add_argument("--legacy-mean-root", action="store_true",
                    help="with --legacy-geometry, also centre the root on the positional mean "
                         "(defect D8b), which is what the original script did. Needed to "
                         "reproduce the pre-fix artifacts exactly.")
    ap.add_argument("--out", default="outputs/multipole_order_sweep.json")
    args = ap.parse_args()

    n_list = args.n_list or [args.n]
    theta_list = args.theta_list or [args.theta]

    dev = torch.device("cpu")   # deterministic; N is small
    torch.manual_seed(args.seed)

    out = {
        "seed": args.seed,
        "softening": args.softening,
        "grid_n": args.grid_n,
        "n_list": n_list,
        "theta_list": theta_list,
        "rows": [],
    }

    for n in n_list:
        p = generate_particle_cloud(n_particles=n, grid_n=args.grid_n, seed=args.seed)
        pos = torch.from_numpy(p["pos"]).float().to(dev)
        mass = torch.from_numpy(p["mass"]).float().to(dev)
        center = pos.mean(0).detach()
        size = float((pos.max(0).values - pos.min(0).values).max().item()) * 1.05

        exact = exact_force(pos, mass, softening=args.softening)
        denom = exact.norm(dim=-1) + 1e-12

        # The tree depends only on n, so build it once and reuse it across theta.
        # Centre and size are no longer supplied: the tree derives a bounding-box root so that a
        # non-uniform cloud cannot escape the root cell (defect D8b). ``size`` is kept as a lower
        # bound to preserve the historical 1.05 padding.
        tree = BarnesHutTree(pos,
                             center=pos.mean(0) if args.legacy_mean_root else None,
                             size=size, max_leaf=8, max_depth=20,
                             legacy_geometry=args.legacy_geometry)
        moments = tree.moments(mass)
        has_quad = bool("quadrupole" in moments and moments["quadrupole"] is not None)
        models = {
            0: HMMP(node_dim=2, hidden_dim=8, n_layers=1, max_leaf=8, max_depth=20,
                    use_quadrupole=False).to(dev),
            1: HMMP(node_dim=2, hidden_dim=8, n_layers=1, max_leaf=8, max_depth=20,
                    use_quadrupole=True).to(dev),
        }

        for theta in theta_list:
            res = {}
            for order in (0, 1):
                total = hmmp_force(models[order], tree, moments, pos, mass, theta, dev,
                                   opening_mode=args.opening_mode)
                rel = (total - exact).norm(dim=-1) / denom
                cos = torch.nn.functional.cosine_similarity(total, exact, dim=-1)
                res[order] = {
                    "rel_err_mean": float(rel.mean().item()),
                    "rel_err_median": float(rel.median().item()),
                    "cosine_mean": float(cos.mean().item()),
                }
            r0 = res[0]["rel_err_mean"]
            r1 = res[1]["rel_err_mean"]
            ratio = (r0 / r1) if r1 > 0 else None
            row = {
                "n": n,
                "theta": theta,
                "p0_rel_err_mean": r0,
                "p0_rel_err_median": res[0]["rel_err_median"],
                "p0_cosine_mean": res[0]["cosine_mean"],
                "p1_rel_err_mean": r1,
                "p1_rel_err_median": res[1]["rel_err_median"],
                "p1_cosine_mean": res[1]["cosine_mean"],
                "ratio_p0_over_p1": ratio,
                "ratio_times_theta": (ratio * theta) if ratio is not None else None,
                "has_quadrupole_moment": has_quad,
            }
            out["rows"].append(row)
            print(f"n={n:>6d} theta={theta:.2f}  p0={r0:.8f}  p1={r1:.8f}  "
                  f"ratio={ratio:.6f}  ratio*theta={row['ratio_times_theta']:.6f}")

    # Consistency check for the prediction ratio ~ 1/theta.
    #
    # A cell is only meaningful if BOTH orders still return an error smaller than the
    # force itself (rel_err < 1). Above some opening angle the COM-based Barnes-Hut
    # criterion starts admitting cells that overlap the query particle, and both orders
    # return nonsense -- that is a genuine breakdown of the scheme, not a quadrupole
    # defect, and it must not be folded into the scaling statistics.
    valid = [r for r in out["rows"]
             if r["p0_rel_err_mean"] < 1.0 and r["p1_rel_err_mean"] < 1.0]
    broken = [r for r in out["rows"] if r not in valid]
    out["n_valid_cells"] = len(valid)
    out["n_broken_cells"] = len(broken)
    out["broken_cells"] = [
        {"n": r["n"], "theta": r["theta"],
         "p0_rel_err_mean": r["p0_rel_err_mean"], "p1_rel_err_mean": r["p1_rel_err_mean"]}
        for r in broken
    ]

    per_theta: dict[float, list[float]] = {}
    for row in valid:
        per_theta.setdefault(row["theta"], []).append(row["ratio_times_theta"])
    out["ratio_times_theta_by_theta"] = {
        f"{t:.4f}": float(np.mean(v)) for t, v in sorted(per_theta.items())
    }
    vals = np.array(sorted(out["ratio_times_theta_by_theta"].values()))
    if vals.size > 0:
        out["ratio_times_theta_mean"] = float(vals.mean())
        out["ratio_times_theta_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out["ratio_times_theta_spread"] = float(vals.max() / vals.min()) if vals.size > 1 else None
        out["ratio_times_theta_rel_scatter"] = (
            float(vals.std(ddof=1) / vals.mean()) if vals.size > 1 else 0.0
        )
    else:
        out["ratio_times_theta_mean"] = None
        out["ratio_times_theta_std"] = None
        out["ratio_times_theta_spread"] = None
        out["ratio_times_theta_rel_scatter"] = None

    ratios = [r["ratio_p0_over_p1"] for r in valid]
    # An empty `valid` list is reachable (a theta where both orders exceed unit relative error), and
    # min()/max() on it raise. Two bugs lived here: this, and below, formatting a None spread.
    out["all_valid_ratios_above_one"] = bool(ratios) and all(r > 1.0 for r in ratios)
    out["min_ratio_p0_over_p1"] = float(min(ratios)) if ratios else None
    out["max_ratio_p0_over_p1"] = float(max(ratios)) if ratios else None

    print(f"\nvalid cells (both orders rel_err < 1): {len(valid)}/{len(out['rows'])}")
    print("ratio * theta (should be ~constant if the R/r scaling of Proposition 3.1 holds):")
    for t, v in out["ratio_times_theta_by_theta"].items():
        print(f"  theta={t}   ratio*theta={v:.6f}")
    if out["ratio_times_theta_mean"] is not None:
        # `spread` is None whenever only ONE theta contributed, which is the default configuration of
        # this script (--theta-list defaults to None, so just --theta is used). The old code guarded
        # `mean` but then formatted `spread` unconditionally and died with
        # "unsupported format string passed to NoneType.__format__" AFTER doing all the work, which
        # is how the first post-D8 campaign lost this sweep.
        spread = out["ratio_times_theta_spread"]
        rel = out["ratio_times_theta_rel_scatter"]
        print(f"  mean = {out['ratio_times_theta_mean']:.6f}  "
              f"std = {out['ratio_times_theta_std']:.6f}  "
              f"rel scatter = {rel:.4%}  "
              f"max/min = " + (f"{spread:.6f}" if spread is not None
                               else "n/a (one theta only -- pass --theta-list for a spread)")
              )
        if spread is None:
            print("  NOTE: a single theta can confirm the ratio but cannot test that ratio*theta is")
            print("        CONSTANT in theta, which is the actual prediction. Use --theta-list.")
    if ratios:
        print(f"all valid ratios > 1: {out['all_valid_ratios_above_one']}  "
              f"(min {out['min_ratio_p0_over_p1']:.6f}, max {out['max_ratio_p0_over_p1']:.6f})")
    else:
        print("all valid ratios > 1: n/a (no valid cells)")
    if broken:
        print(f"breakdown cells (excluded): "
              + ", ".join(f"n={r['n']},theta={r['theta']}" for r in broken))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
