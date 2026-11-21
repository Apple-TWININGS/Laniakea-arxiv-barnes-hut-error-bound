# Provenance index

Every quantitative claim in `paper.tex` mapped to the artifact that produced it, plus the
launch command and the run duration wherever those were recoverable.

**Why this file exists.** During the ICML conversion a number in the paper — the "hidden=128
result" `0.025891` in the capacity paragraph of Section 5 — turned out to have **no artifact
anywhere**. It had been written into `paper.tex` as prose, and because `train_real_dist.py`
recorded only metrics and not its configuration, no finished run could be attributed to a hidden
width after the fact. Finding it required an exhaustive search of two GPU instances, their shell
histories and 147 result JSONs, and it still could not be traced. This index is the fix: a number
in the paper either maps to a file here or it does not belong in the paper.

Related fix: `train_real_dist.py` and `train_real.py` now write a `config` dict and a
`config_id` string into every result JSON, so a run can never again become unattributable.

---

## 1. Log ↔ artifact index

The raw training logs are archived beside this file. Wall-clock is from the log's first torchrun
timestamp to the file's last write.

| Artifact | Log | World size | Epochs | Result | Wall-clock |
|---|---|---|---|---|---|
| `outputs/dist_2ep.json` | (not archived) | 4 | 2 | 0.508186 | — |
| `outputs/dist_20ep.json` | (not archived) | 4 | 20 | 0.040763 | — |
| `outputs/dist_100ep.json` | `run_log.txt` | 4 | 100 | 0.026093 | 2026-08-26 19:11:59 → 19:30 (~18 min) |
| `outputs/dist_h256.json` | `dist_h256.log` | 4 | 100 | 0.025888 | 2026-09-03 12:27:32 → 12:49 (~22 min) |
| `outputs/dist_6card_100ep.json` | `dist_100.log` | **6** | 100 | 0.025359 | 2026-08-26 23:26:03 → 23:40 |
| `outputs/dist_phys.json` | `dist_phys.log` | 4 | 200 | 0.025364 | 2026-08-31 |
| `outputs/dist_phys_pk_seed{0,1,2}.json` | `dist_phys_pk_seed{0,1,2}.log` | 4 | 200 | 0.022372 / 0.023005 / 0.022462 | 2026-09-03 |
| `outputs/hmmp_pure_200ep{,_seed1,_seed2}.json` | `hmmp_pure_200ep{,_seed1,_seed2}.log` | 1 | 200 | 0.023769 / 0.023421 / 0.022874 | 2026-09-06/07 |
| `outputs/tr_2M3_cos.json` | `tr_2M3_cos.log` | 1 | 100 | 0.025244 | 2026-08-26 21:38 |
| `outputs/tr_2M3_15M.json` **(not retrieved)** | `outputs/tr_2M3_15M.log` (failed attempt; see below) | 1 | 100 | not recorded here — the run was reported to have produced a result, which was never pulled into the repository | 2026-08-26 21:40 |
| `outputs/tr{0,1,2,3}.json` | `tr{0,1,2,3}.log` | 1 | 100 | 0.024868 / 0.024776 / 0.025449 / 0.024901 | 2026-08-26 18:54 |
| `outputs/dist_phys_weighted.json` | `weighted.log` | 4 | 200 | 0.022965 | 2026-09-11 14:36 |
| `outputs/{matched,1M_loop,500k_loop}.log` | — | 1 | 100 | baseline campaigns | 2026-09-11 |
| `outputs/dist_h128_100ep{,_seed1,_seed2}.json` | `dist_h128{,_seed1,_seed2}.log` | 4 | 100 | 0.025419 / 0.024736 / 0.025065 | 2026-09-12 14:43 → 15:49 |

**`tr_2M3_15M`: one recovered log is a failed attempt, and that is all it proves.** The row above used
to read "(on instance only, not pulled)", i.e. *the file exists on an instance and was never copied*.
A log was recovered from a persistent disk on 2026-09-25 showing `nohup` start,
`[real] loaded contract npz: 2300000 particles`, then on the first backward pass
`torch.cuda.OutOfMemoryError: Tried to allocate 98.00 MiB. GPU 0 has a total capacty of 79.14 GiB of
which 72.75 MiB is free. Process 995008 has 79.06 GiB memory in use.`

**That log documents one failed attempt. It does not establish that the run never succeeded, and this
file previously drew that conclusion and had to withdraw it.** The author's record is that the
`tr_2M3_15M` run did complete on one of several rental instances and that its result was real rather
than a placeholder — the instance was simply never identified, and the artifact was never pulled in.
An OOM log on one box is consistent with a later attempt on another box having succeeded; with several
instances in rotation there is no reason to treat a single log as the whole history.

So the honest state is: **the artifact is not in the repository and has not been located**, the value
printed in this row is therefore blank rather than zero, and the failed attempt's log is committed at
`outputs/tr_2M3_15M.log` as what it is. Nothing in the manuscript cites this run — it is a
repository-level record only — but "not located" is the strongest claim this file is entitled to make.

### Withdrawn runs, kept for the record

The row below belongs to the withdrawn density-reconstruction experiments (§3.0), not to any
current claim. It is kept because it is the only **properly powered, fully configured** run at the
largest scale, and because it settles the question that defect §3.1 left open.

**Full-capacity 2.3M, hidden=128, 3 layers, 100 epochs, three seeds: val MSE
`0.025074 ± 0.000342`** (per-seed `0.025419` / `0.024736` / `0.025065`). Every run recorded
`config_id = n2300000_h128_l3_e100_mse`, so unlike the earlier rows this one *is* fully
attributable — the config-recording fix from §3.1 is what makes it so. Its standard deviation is
`1.4%` of its mean, i.e. the seed spread at 2.3M is small once the width is fixed, which also
explains why the single-seed `0.025888` (hidden=256) and the unattributable `0.025891` could not be
compared meaningfully.

A caveat that travels with every row above except where noted: **the log records
`world_size`, `epochs` and `n_particles`, but not `hidden` or `layers`.** The hidden width is
known only where it survives in the *filename* (`dist_h256`) or in a retained launch script or
shell history entry. This is the defect described in the header.

### Status of every artifact family (added 2026-09-12)

The two halves of this repository now have very different standing and must not be mixed. Every
number in `paper.tex` comes from a row marked **LIVE**.

| Family | Count | Status | Why |
|---|---|---|---|
| `order_sweep_tidal_*`, `multipole_order_sweep{,_tidal}`, `e1_*`, `e2_*`, `e4_*`, `agg_*`, `cell_geometry_{constant,invariances}`, `tidal_*_diagnostic`, `misclassification_vs_environment`, `determinism_*`, `exact_sampled_fixed_*`, `exact_nbody.json`, `scaling.json`, `memory_*` | 45 | **LIVE** | produced or re-produced after every known defect. These are the paper's numbers; nothing here needs re-running |
| `sampled_validation/exact_sampled_n*.json`, `thetasweep_th*_seed*.json`, `diag_farfield.json` | 22 | **SUPERSEDED / INVALID** | pre-R5 tidal tensor (not the Jacobian of the force), and for `diag_farfield` the retired leaf-only cell expansion (R2). Kept deliberately beside the corrected files so the before-state survives |
| `dist_*`, `hmmp_pure_*`, `tr_*`, `ens*`, `baseline_*`, `cross_astrid_*`, `real_*`, `single_*`, `pure*`, `scale_*`, `probe_*`, `ablate_*`, `lrsweep*`, `eval_phys_*`, `phys_*.pt`, `pure_*.pt`, `vis_*.png`, `matched.log`, `1M_*`, `500k_*` | 148 | **WITHDRAWN — do not cite any MSE** | the density-reconstruction campaign. Its node features contained the regression target (§3.0), so every comparison among models is void. The training scripts, the `config` recording and the ensemble machinery stay reusable; the numbers do not |
| the leak-free control probes (`probe_legit_100k` + five matched 100k seeds, §3.0) | 6 | **LIVE, artifacts in `docs/provenance/leakfree_control/`**. The MSE ordering is **not stable**: the probe has Graph Transformer ahead (`0.025726` vs `0.027378`), and the matched seeds mix by which test statistic is used. No directional claim; no recorded leak-free configuration wins both MSE and the power spectrum | downstream measurement after the feature fix; the pre-fix 2026-10-19 run is withdrawn under `docs/provenance/withdrawn_pre_fix/` |
| the 2026-10-20 leak-free run set (100K 3 seeds, 500K/1M 2 seeds, 30-ep rerun, A/B leak-free, ceiling) | 12 | **LIVE, artifacts in `docs/provenance/leakfree_control/2026-10-20/`**. 100K means flip by statistic; 500K/1M HMMP wins MSE but loses the power spectrum (12× vs GT at 500K; 466× vs ClusterGCN at 1M; GT OOM at 1M). No stable winner; no configuration wins both | fresh rerun after the fix; see the folder README for commands, hashes and the training-free kNN ceiling |

Two things inside the withdrawn block are still usable as evidence rather than as results:

* the full-capacity 2.3M / 100-epoch / three-seed run (`dist_h128_100ep*`, `0.025074 ± 0.000342`) is
  fully attributable, and its **seed-spread** statement survives even though its MSE is not
  comparable to anything;
* `graph_cache/` (4 files, 1.1 GB) is ours and is backed up; it is a regenerable cache, not data.

Re-running the withdrawn campaign is possible and would convert it from *invalid* to *valid*, but it
would not restore the claim. The reason is now stronger than the leak: §3.0b shows the target is a
deterministic function of the input positions. The leak-free comparisons do not restore a ranking
either: the committed probe has Graph Transformer ahead on MSE, and the five matched 100k seeds mix
by statistic. (The 2026-10-19 30-epoch console run that briefly appeared here came from the pre-fix
`Laniakea-HMMP` copy, whose `graph.py` still used `log1p(true_density)`; it is withdrawn under
`docs/provenance/withdrawn_pre_fix/` and is not leak-free evidence.) Where the operator wins MSE at
larger `N` it loses the power spectrum by five to twelve times. No recorded configuration wins both,
and no directional claim is drawn from the control. Any such re-run is a different paper's decision,
not this one's.

**Not ours, deliberately left alone.** The GPU box carries an 11 GB `checkpoints/` directory holding
GLUE-task weights (`cola/`, `mnli/`, `mrpc/`, ..., files named `dac_mrpc_best.pt`) from another
project sharing the machine, plus a 15 GB `hf/` model cache. Neither was copied into this project's
backup. Only `graph_cache/` and the box's `outputs/` belonging to Laniakea-HMMP were taken.


---

## 2. Paper number → artifact

The paper was rescoped after the target-leak defect in §3.0 was found. The density-reconstruction
results are **withdrawn**, and the numbers below are the only quantitative claims in the current
`paper.tex`.

### Section 4.1 — agreement with exact N-body forces

| Number | Artifact |
|---|---|
| global relative error `0.001996`, cosine `0.999998` (N=2000, θ=0.5, ε=1e-4, **post-D8**, synthetic lognormal) | `docs/provenance/artifacts/exact_nbody_fixed.json` (`scripts/exact_nbody.py --opening-mode com`) |
| the same measurement **pre-D8** (`--legacy-geometry`): `0.021419`, cosine `0.999771` | **artifact not retained** — see the note below and `docs/RERUN_MISSING_ARTIFACTS.md` |
| exact sum `0.0795` s, Barnes–Hut operator `0.0758` s (one CPU core, single run — noisy) | `docs/provenance/artifacts/exact_nbody_fixed.json` (`exact_time_s` / `bh_time_s`) |
| far field alone: per-particle mean rel err `0.041258`, mean cosine `0.999254` | `outputs/_psweep_fixed_n2000.json` (the `p=0` row) |

**Two repair notes (2026-09-25), because both rows used to point somewhere a reader could not go.**
Both the post-D8 rows named `outputs/exact_nbody_fixed.json`, which **does not exist**: `outputs/` is
gitignored and only partially tracked, so the run was never committed under that path. The file that
holds the quoted values — `complete_operator_rel_error = 0.0019956084326449497`, cosine
`0.9999982118606567`, `exact_time_s = 0.07945489883422852`, `bh_time_s = 0.07579708099365234`, hence
`0.0795` s and `0.0758` s — is the committed copy
`docs/provenance/artifacts/exact_nbody_fixed.json`, and the rows now name it. This is the defect the
README's pre-submission list carried as "names an artifact that is not committed"; it is fixed at the
source here.

The **pre-D8 row is a different matter and is left open deliberately**: no file holds `0.021419` /
`0.999771`. The nearest surviving artifact, `outputs/exact_nbody.json`, is an older and different run
(`far_field_rel_error = 0.019234766448254722`, cosine `0.9998151063919067`), so it cannot be relabelled
as this row. The number stays because it is quoted in the paper's before/after pair for D8, but its
provenance is now marked **not retained** rather than implied by a dead path.

The artifact keys `far_field_rel_error` / `far_field_cosine` are **misnamed**: the script computes
them from `far + leaf`, i.e. the *complete* operator, and says so in a comment. See §3.3.

The two rows above measure different quantities — a global field-norm ratio for the complete
operator versus a per-particle mean for the far field — and come from different scripts. The paper
says so at the point of use; they must not be presented as a like-for-like ablation.

### Section 4.2 — multipole-order validation

| Number | Artifact |
|---|---|
| `2.374879 ± 0.117030`, relative scatter `4.93%`, over 21 settings, **post-D8**, `--opening-mode com` | `docs/provenance/artifacts/multipole_order_sweep_fixed.json` |
| the same grid **pre-D8** (`--legacy-geometry --legacy-mean-root --opening-mode center`): `1.171830 ± 0.059440`, scatter `5.07%` | `docs/provenance/artifacts/multipole_order_sweep_original_repro.json` |
| the 21-cell `ratio × θ` grid (Table `tab:ordersweep`) | the `_fixed` file, `rows[]` filtered to θ ≤ 0.60 |
| tidal tensor, same grid, post-D8: `1.773758 ± 0.050710`, scatter `2.86%`; per-θ table in Appendix `app:percell` | `docs/provenance/artifacts/multipole_order_sweep_tidal_fixed.json` (`1.7737576488877143`; a previous pass reported this file as not holding the value, which was a string-match error — it was compared against the *rounded* `1.773758` instead of the stored float) |
| θ-power of the leading error: force `2.98`, tidal `3.39` (classical `3.0`) | same file |

> **The artifact this section originally cited, `outputs/multipole_order_sweep.json`, was never
> committed** and is absent from the repository — only its tidal sibling is tracked. The number it
> carried is nevertheless reproduced by the pre-D8 flags: `1.171830` against the reported
> `1.171843`, agreeing to six significant figures. (The scatter statistic differs slightly,
> `5.07%` against `5.5174%`, which points at a different std convention.) Either re-commit the
> original or regenerate it with `--legacy-geometry --legacy-mean-root`; nothing is lost either way,
> but a paper whose stated contract is "every number maps to a committed artifact" cannot leave this
> gap open.

Reproduce the headline summary straight from the **post-D8** artifact:

```bash
python -c "
import json, statistics as st
d = json.load(open('outputs/multipole_order_sweep_fixed.json'))
v = [r['ratio_times_theta'] for r in d['rows'] if r['theta'] <= 0.6001]
m, s = st.mean(v), st.stdev(v)
print(len(v), '%.6f %.6f %.4f%%' % (m, s, 100*s/m))
"
# -> 21 2.374879 0.117030 4.9278%
```

**The grid in Table `tab:ordersweep` is regenerated from this file.** Note that the version printed
in earlier drafts did *not* match it in the fifth and sixth decimal (e.g. `1.229750` against
`1.229770`) even though the summary statistics matched exactly — i.e. the summary was computed from
the artifact and the grid transcribed from an earlier run of it. See §3.3.

### Section 4.2b — the multipole-order sweep extended to the tidal tensor (added 2026-09-12)

| Number | Artifact |
|---|---|
| `ratio·θ` for the **force**, θ ≤ 0.5, all three N: `1.192`, relative scatter `2.4%` | `outputs/multipole_order_sweep_tidal.json` (`ratio_rel_times_theta_*_force`) |
| the force column reproduces `outputs/multipole_order_sweep.json` exactly (same `ratio·θ` per cell, e.g. n=2000/θ=0.20 `1.2007`) | compare the two files' `rows[]` |
| `ratio·θ` for the **tidal tensor**: `0.860`, relative scatter `4.2%` over θ ≤ 0.5 | same file (`ratio_rel_times_theta_*_tidal`) |
| per-θ tidal `ratio·θ`: `0.914 / 0.882 / 0.868 / 0.844 / 0.839 / 0.811` (θ = 0.15…0.50) | same file, `ratio_rel_times_theta_by_theta_tidal` |
| θ-power of the p=0 error over θ ≤ 0.5: force `3.23`, tidal `3.56` | same file, `theta_power_rel_*` |
| tidal/force `mean_rel_err` at θ=0.5: `3.52` (N=32000), `3.20` (N=8000), `2.97` (N=2000) | same file, `p{0,1}_{force,tidal}_rel_err_mean` |
| absolute-error ratio is outlier-dominated: 20 of 32,000 nodes carry `95.4%` of the p=1 tidal error at θ=0.15, where `err(p=0)/err(p=1) = 1.008` | `outputs/tidal_error_norm_diagnostic.json` |
| float32 cancellation floor is 6–7 orders below those errors (median `4.3e-6` vs `7.1e-4` at N=32000/θ=0.15) | same file |
| `surface` opening criterion cuts the top-20 share at θ=0.5 from `61%` to `35%` (and raises the absolute ratio `1.38 → 2.17`) | `outputs/tidal_error_norm_diagnostic_surface.json` |
| the pre-fix tensor differs from the correct one by `1280.5×` and is `86.1%` isotropic (correct: `0.0000`) | `outputs/tidal_convention_diagnostic.json` |

Reproduce:

```bash
python scripts/sweep_multipole_order_tidal.py --out outputs/multipole_order_sweep_tidal.json   # 210 s, CPU
python scripts/diagnose_tidal_convention.py
python scripts/diagnose_tidal_error_norm.py --n 32000 --thetas 0.15 0.5
python scripts/diagnose_tidal_error_norm.py --n 32000 --thetas 0.15 0.5 --opening-mode surface \
    --out outputs/tidal_error_norm_diagnostic_surface.json
```

**Every tidal number predating this section is invalid — see §3.4.**

### Section 4.2c — the corrected section 1a validation at 2.3M (added 2026-09-12, GPU box)

Re-run of the sampled exact-N-body ladder with the corrected tidal tensor. The force columns
**reproduce the originally published values** (at the quoted precision — see the determinism row at
the end of this section; the GPU path is not bit-deterministic), which is what makes the corrected
tidal columns admissible.

| Number | Artifact |
|---|---|
| force column unchanged at all five (N, Q) cells, e.g. `0.024071` at 2.3M/2048 | `docs/provenance/sampled_validation/exact_sampled_fixed_n*_q*.json` |
| corrected tidal `mean_rel_err`, monopole far field: `0.124150 / 0.127748 / 0.139563 / 0.136239 / 0.140502` | same files |
| complete `p=1` operator: force `0.016028`, tidal `0.100757` at 2.3M/2048 (also 10k/100k/1M/2.3M@1024) | `docs/provenance/sampled_validation/exact_sampled_fixed_quad_n*_q*.json` |
| tidal/force error ratio `4.47 / 4.96 / 5.45 / 5.47 / 5.84` (p=0) and `5.21 / 5.53 / 5.98 / 6.13 / 6.29` (p=1) | same files, `tidal.mean_rel_err / force.mean_rel_err` |
| the superseded (invalid) values, kept for the record: `0.051373 / 0.049680 / 0.049167 / 0.049605 / 0.050200` | `docs/provenance/sampled_validation/exact_sampled_n*.json` — the pre-fix artifacts, already committed before the defect was found, so the before/after pair sits in one directory |

Run logs for the corrected ladder are `docs/provenance/exact_sampled_fixed_*.log` (one per run,
written by `scripts/rerun_sampled_validation.sh`, 65 s for the whole ladder); the box-side copies
live in `/root/autodl-tmp/outputs/`.

### Section 4.2d — is the order-shift constant universal? (added 2026-09-12)

`scripts/sweep_multipole_order_tidal_camels.py` repeats the order sweep on real CAMELS data
(N=100000, Q=2048) with the same operator as §1a. Answer: the shift is robust, the constant is not.

| Result | Artifact |
|---|---|
| synthetic, sampled pipeline: force `1.219 ± 6.4%`, tidal `0.819 ± 5.4%` | `outputs/order_sweep_tidal_synth100k_leaf8.json` |
| synthetic, equal masses: force `1.153 ± 12.5%`, tidal `0.787 ± 7.6%` | `outputs/order_sweep_tidal_synth100k_eqm_leaf8.json` |
| **CAMELS**: force drifts `1.686 → 0.776` (mean `1.208 ± 31.7%`), tidal `0.902 → 0.710` (mean `0.797 ± 8.3%`) | `outputs/order_sweep_tidal_camels_n100000_leaf8.json` |
| CAMELS, `max_leaf=16`: force `1.207 ± 32.5%`, tidal `0.755 ± 6.4%` (leaf size is not the cause) | `outputs/order_sweep_tidal_camels_n100000_leaf16pad.json` |
| CAMELS, `size_pad=1.0` (same data): force `1.037 ± 24.7%` vs `1.208 ± 31.7%` at `pad=1.05` | `outputs/order_sweep_tidal_camels_n100000_leaf8pad1.json`, `..._leaf8.json` |

Every one of those has a `.log` beside it with the run's stdout (`docs/provenance/order_sweep_tidal_*.log`),
from a logged re-run that reproduced the un-logged JSONs exactly — the seeds are deterministic, so
the log is a check on the artifact rather than a separate result.

**Post-D8 re-measurement, three scales (2026-09-17).** The rows above are **pre-D8** and are kept as
the before/after pair; the paper now quotes the block below. Both the mean *and* the scatter moved,
because the geometry fix admits a different set of cells.

| N | force product | force scatter | force range | tidal product | tidal scatter | tidal range | artifact |
|---|---|---|---|---|---|---|---|
| 100,000 | 3.249 | 15.7% | 1.64× | 1.758 | 3.9% | 1.10× | `order_sweep_tidal_camels_n100000_postd8.json` |
| 1,000,000 | 3.743 | 24.4% | 1.91× | 1.351 | 35.7% | 4.09× | `order_sweep_tidal_camels_n1000000_postd8.json` |
| 2,300,000 | 3.318 | 23.5% | 1.86× | 1.121 | 53.9% | 8.10× | `order_sweep_tidal_camels_n2300000_postd8.json` |

These are the artifacts' own summary fields (`ratio_times_theta_mean_*`,
`ratio_times_theta_rel_scatter_*`); scatter is the sample standard deviation over the seven `theta`
values, relative to the mean, and "range" is max/min of the product. Both files live in
`docs/provenance/artifacts/`.

**This is where the paper's stale `2.2×` came from.** The pre-D8 100k row drifts `1.686 → 0.776`,
i.e. `2.17×`; that number was in `tab:ordersweep`'s caption, §"What is verified, and what is not" and
Remark B.1 long after D8 was fixed. Post-D8 the same drift is `1.45×`, and at the three scales it is
`1.64 / 1.91 / 1.86×` — not identical, which the pre-D8 claim also asserted. The paper now states the
post-D8 values and no longer claims a single drift factor.

**And the substantive finding behind the rewrite:** the *tidal* product's scatter **grows with `N`**
(`3.9% → 35.7% → 53.9%`) while the force's does not (`15.7 → 24.4 → 23.5%`), so the ordering between
the two derivatives inverts with scale — the tidal product is the *more* stable of the two at `10^5`
and the *less* stable one at `2.3M`. At `2.3M` the tidal product moves over an `8.1×` range, and at
the tightest opening it is `0.208` (quadrupole worth `1.4×`) against `1.816` at `10^5` (`12.1×`): the
tidal order shift is not merely noisier at scale, it is **weaker**. What survives at every scale is
the **sign**: the `p=0`/`p=1` ratio exceeds one in all 21 real-data configurations for both the force
(`4.0–24.5`) and the tidal tensor (`1.4–12.1`). `paper.tex` §4.5 (`sec:order`) and the Conclusion now
say exactly that, and `tab:camelsorder` carries the table.

### Section 4.2e — is the drifting constant an aggregate or a per-interaction quantity? (added 2026-09-12)

Answering the objection that a bound is built per interaction while everything we measure is an
aggregate. `scripts/diagnose_aggregate_vs_percell.py` evaluates the same interaction list at three
levels of aggregation, with every cell expanded to **all** of its descendant particles.

| Result | Artifact |
|---|---|
| synthetic $N=100{,}000$: per-interaction `1.169 ± 0.7%`, coherent aggregate `1.193 ± 4.0%`, per-node mean `1.275 ± 3.7%` — all three agree and are constant | `outputs/agg_vs_percell_synth_n100000.json` |
| CAMELS $N=100{,}000$: per-interaction `1.066 ± 12.4%` (drifts `1.21 → 0.93`), aggregate `1.037 ± 22.4%` (`1.34 → 0.81`), per-node mean `1.040 ± 22.8%` (`1.36 → 0.80`) | `outputs/agg_vs_percell_camels_n100000.json` |
| so: the drift is present in the summed per-interaction remainders too (not a pure coherence artefact), but is amplified by the coherent sum (`×1.30` vs `×1.65`) | same files |

**This is why the paper claims only that the measured product is data-dependent and must be quoted
per dataset.** It does *not* claim that no universal constant exists: every quantity measurable here
is a weighted average over the admitted-cell distribution, whereas a bound's constant is a statement
about one cell geometry.

### Section 4.3 and Appendix H — the review-response experiments (added 2026-09-12, GPU box)

Three experiments a referee asked for, all on the real snapshot.

**E1 — the acceptance test as a fix** (`scripts/sweep_multipole_order_tidal_camels.py
--opening-mode surface`). The centre-of-mass test collapses at $\theta=0.70$; the
distance-to-cell test does not, and at *matched interaction count* is the more accurate of the two:

| Number | Artifact |
|---|---|
| com: force `2.59e-2 / 5.79e-2 / 8.73e0 / 1.20e1` and tidal `1.43e-1 / 2.88e-1 / 6.80e7 / 1.47e8` at θ = 0.50/0.60/0.70/0.80 | `outputs/e1_camels_n100000_center_bigtheta.json` |
| cell: force `7.70e-3 / 1.09e-2 / 1.50e-2 / 2.02e-2` and tidal `3.47e-2 / 5.69e-2 / 8.11e-2 / 1.14e-1`, same θ | `outputs/e1_camels_n100000_surface_bigtheta.json` |
| far entries per query node, com `102/66/44/28` against cell `251/183/143/117` | both files, `rows[].n_far_entries` |

**E2 — the physical metrics** (`scripts/eval_tidal_eigenmetrics.py`). Eigenvalue error, principal-axis
misalignment (restricted to nodes whose top two $|\lambda|$ differ by ≥1.5×, i.e. 30–40% of them) and
the Zel'dovich class-flip rate:

| Number | Artifact |
|---|---|
| 2.3M/θ=0.5/com: class flip `6.67%` (p=0), `5.27%` (p=1); p95 axis `6.09°`/`4.76°` | `outputs/e2_eigen_camels_n2300000_center.json` |
| 100k, com: flip `0.71% / 5.27% / 22.75%` at θ `0.3 / 0.5 / 0.7` (p=0) | `outputs/e2_eigen_camels_n100000_center.json` |
| 100k, cell: flip `1.22% / 2.81%` at θ `0.5 / 0.7` (p=0), `2.15%` at θ=0.7 (p=1) | `outputs/e2_eigen_camels_n100000_surface.json` |
| exact class distribution: `71%` one positive eigenvalue (sheet), `29%` two (filament) | same files |

**E3 — the gradient a field-level reconstruction actually uses** (`scripts/eval_vjp_error.py`).
The paper sells differentiability; this measures it, by differentiating a scalar with a cosmological
reading — the density-weighted tidal energy $L(w)=\sum_i w_i\|T_i\|_F^2$ — through both the exact
field and the operator, and comparing the two vector-Jacobian products. A VJP contracts the per-node
Jacobians with an adjoint vector and *sums*, so it can either amplify the per-node tidal error
coherently or cancel it. The **amplification factor** is $\|\hat g-g\|/\|g\|$ over the
$\|T\|^2$-weighted per-node tidal error **at the same multipole order**.

Synthetic cloud (`data_cache/synthetic_n200000.npz`, built by `scripts/make_synthetic_npz.py` with
`--n 200000 --grid-n 128`, seed 0), $n=10^5$, $Q=256$, 15 seeds per cell, density-weighted $L$, $p{=}0$:

**Its identity is pinned here in full, and the generator was confirmed to reproduce it** (2026-09-25).
The file is not committed — `data_cache/` is gitignored and has zero tracked files, because it is a
*generated input*, not an artifact — so the only thing that makes the table above checkable is the
content hash. It used to be written as `pos_sha256 84dcab6c…`, eight hex characters of sixty-four,
which cannot verify a rebuild. Rebuilding with the documented command produced:

| | value |
|---|---|
| `pos_sha256` (sha256 of `pos.tobytes() + mass.tobytes()`) | `84dcab6c14c05d9d905e8e2a729956c5bda662e9566613a6c3ee432f010570ef` |
| `.npz` file sha256 | `490FD97ADB8AA08AE41EC75AA621A1A651943E83848DCDF23D97535A3CBDC0A3` |
| `mass_mean` / `mass_std` | `1.0` / `0.6565225124359131` (the "0.657" quoted elsewhere in this file) |

The first eight characters are the ones this file used to truncate to, so **the regeneration is
byte-identical to the original particle set** — the truncated hash was not mismatched, merely
unverifiable, and that is now fixed. Two further facts, both tested rather than assumed: the file hash
is reproducible across regenerations, because `numpy.savez` stamps its zip entries with a fixed
`1980-01-01` rather than wall-clock time (verified by inspecting the archive, and by regenerating across
a time boundary); and the E3 sweep therefore has a rebuildable input with a checkable identity, which is
what a re-run of that sweep would need.

| θ | criterion | grad rel err (median / max) | cos (min) | amp (median / max) | amp>1 | blow-ups / 15 |
|---|---|---|---|---|---|---|
| 0.5 | center | `1.31e-3` / `1.34e-2` | `0.999914` | `0.172` / `0.645` | 0 | 0 |
| 0.6 | center | `5.60e-3` / `3.91e-2` | `0.999236` | `0.294` / `0.586` | 0 | 0 |
| 0.7 | center | `8.04e-2` / **`inf`** | **`0.000000`** | `0.590` / **`inf`** | 3 | **5** |
| 0.5 | surface | `3.99e-4` / `3.13e-3` | `0.999996` | `0.175` / `0.512` | 0 | 0 |
| 0.6 | surface | `5.19e-4` / `7.37e-3` | `0.999981` | `0.193` / `0.613` | 0 | 0 |
| 0.7 | surface | `7.52e-4` / `7.66e-3` | `0.999979` | `0.167` / `0.510` | 0 | 0 |

Three results:

1. **In the valid regime the VJP *contracts*; it does not amplify.** In every cell at θ ≤ 0.60, and
   in every `surface` cell at every θ, the amplification is `< 1` (max `0.714`). The gradient is
   **more** accurate than the tensor it came from, because the per-node errors partially cancel in
   the adjoint sum. The paper's per-node tidal error is therefore a *conservative* statement about
   the gradient, which is the direction that favours it. The θ=0.70 com group is the sole
   exception, and there the failure is not marginal: its finite amplification reaches `28.9` on top
   of the divergent runs.
2. **The θ=0.70 centre-of-mass breakdown propagates to the gradient, catastrophically and
   intermittently.** 5 of 15 seeds blow up (per-node weighted tidal error > 1), reaching `9.78e8`
   ($p{=}0$) and `2.07e14` ($p{=}1$); the gradient error hits `inf`/`nan` and the cosine collapses
   to `0.000000`. The mechanism is the known one: a query node that lands on or beside the centre of
   mass of an admitted far cell gives $r^2=|COM-x|^2+10^{-8}\to10^{-8}$, hence $1/r^3\to10^{12}$.
   This is a **silent** failure — θ=0.5 and 0.6 are clean on the same code path, so no single
   well-behaved run warns that the acceptance test is unsafe.
3. **The distance-to-cell criterion removes the failure entirely.** Over the same 15 seeds
   `surface` has `0/15` blow-ups at every θ; at θ=0.70 it is `107×` better at the median and
   `6.5e10×` better in the worst case (the com worst case is in fact unbounded — its gradient
   error reaches `inf`), and at θ=0.5 — the paper's operating point — it is already `3.3×`
   better. It is never the worse choice in this sweep. Where the com criterion does *not* blow up
   at θ=0.70 it still reaches a finite amplification of `28.9`, i.e. the failure is not confined
   to the divergent runs.

| Number | Artifact |
|---|---|
| Full 90-run sweep (2 criteria × 3 θ × 15 seeds) | `outputs/e3_sweep/s{0..14}_th{0.5,0.6,0.7}_{center,surface}.json` |
| p=0 summary table above | `docs/provenance/artifacts/e3_sweep_summary_p0.json` |
| p=1 summary (same pattern: center `5/15` blow-ups, `2.07e14` worst; surface `0/15`) | `outputs/e3_sweep_summary_p1.json` |
| equal-mass control at θ=0.7 (center `3.93e-1`, cos `0.919`, amp `1.290` vs surface `9.72e-5`, cos `1.000000`, amp `0.613`) | `outputs/vjp_e3_eqm_n100000_th0.7_{center,surface}.json` |

**Defect found and fixed in this script on its first-ever run.** The per-node tidal error was
computed once with `use_quadrupole=False` *outside* the order loop and reused for both orders, so the
$p{=}1$ amplification divided the $p{=}1$ gradient error by the $p{=}0$ tensor error. This
**understated** the $p{=}1$ amplification: at $n=10^5$, θ=0.5 it read `0.389` where the same-order
ratio is `0.651`. The before-state is kept as `outputs/vjp_n100000_prefix_bug.json`. `--opening-mode`
was also added, so the breakdown can be reported for the gradient and not only for the tensor.

**Scope caveat.** Every number above is on the **synthetic** cloud, not the real CAMELS snapshot
(`data_cache/` is not in the repository). The synthetic cloud is clustered with density-proportional
masses (`mass_std = 0.657`), whereas the real subsample is equal-mass; the equal-mass control in the
table reproduces the pattern, so the qualitative result does not depend on the mass spectrum.

**Reproduce:**
```bash
python scripts/make_synthetic_npz.py --n 200000 --grid-n 128 --out data_cache/synthetic_n200000.npz
for m in center surface; do for th in 0.5 0.6 0.7; do for s in $(seq 0 14); do
  python scripts/eval_vjp_error.py --npz data_cache/synthetic_n200000.npz --n 100000 \
    --n-query 256 --theta $th --seed $s --opening-mode $m \
    --out outputs/e3_sweep/s${s}_th${th}_${m}.json
done; done; done
python scripts/summarize_e3.py --dir outputs/e3_sweep --order 0 --out outputs/e3_sweep_summary_p0.json
```

**E4 — query-sampling error bars** (`--query-seed`, particle set and tree held fixed):

| Number | Artifact |
|---|---|
| p=0: force `0.02434 ± 0.00029` (1.2%), tidal `0.1381 ± 0.0024` (1.7%) over five draws | `outputs/e4_seed{0..4}_fixed_n2300000_q2048.json` |
| p=1: force `0.01601 ± 0.00027`, tidal `0.1004 ± 0.0012` | `outputs/e4_seed{0..4}_fixed_quad_n2300000_q2048.json` |

**These ten artifacts were recovered on 2026-09-25 from the run logs, not re-measured.** They were
absent from the repository, and re-running them would have produced *new* numbers (the same statistics
are platform- and query-sample-sensitive — see the determinism note below), which is not the same thing
as recovering the record. It turns out no re-run was needed: each of these scripts **prints its JSON
and then writes the same JSON to disk**, so the artifact was sitting in the log the whole time:

```
{ "n": 2300000, ... , "tidal": { ... } }
wrote /root/autodl-tmp/outputs/e4_seed0_fixed_n2300000_q2048.json
```

The 53 run logs are archived outside the repository at `_withdrawn_logs_Laniakea_20260917/` (indexed by
`docs/EXTERNAL_LOGS.md`; the directory name refers to logs kept *outside* the repository, not to
withdrawn experiments — `EXTERNAL_LOGS.md` is the authority on which of the 53 are usable). Ten of them
carry an `e4` payload; `scripts/recover_from_logs.py` parses the payload and rewrites the artifact.

Two independent checks stand behind this, both of which had to pass before the files were adopted:

1. **The parser is validated against artifacts that already exist.** Four `exact_sampled_*` logs in the
   same archive have their outputs committed under `docs/provenance/sampled_validation/`; the recovered
   payloads match those files **field for field**, so the log text is byte-faithful, not a paraphrase.
2. **The recovered set reproduces the numbers quoted three lines above, exactly.** Aggregating the five
   recovered `p=0` draws gives force `0.02434 ± 0.00029` and tidal `0.13812 ± 0.00236`; the `p=1` set
   gives `0.01601 ± 0.00026` and `0.10041 ± 0.00118`. That is the documented table to every printed
   digit, which a wrong or truncated parse could not have produced.

That the recovered files are the **post-D8** runs is checkable and was checked: they carry `_fixed_` in
the name, and the `seed0` payload has `interaction_far_entries = 302644` / `near = 8978`, identical to
the committed post-D8 reference `exact_sampled_fixed_n2300000_q2048.json` — the same corrected tree.
(The other seeds differ slightly in those counts because a different query seed admits a different set
of far cells, which is the experiment's design, not a different geometry.)

The recovered JSONs and their ten logs are both tracked, so the record and its evidence travel
together.

**Determinism — a claim we had to withdraw.** `scripts/check_gpu_determinism.sh` runs the identical
2.3M configuration three times:

| Number | Artifact |
|---|---|
| p=0 agrees to `~1e-7` relative (float32 round-off); p=1 varies at `~0.4%` | `outputs/determinism_p{0,1}_run{1,2,3}.json` |

The p=1 spread is not instability in the quadrupole path: the error being measured is ~3× smaller, so
the same field-level noise is a larger fraction of it. Both are below the query-sampling spread above,
which is the uncertainty the *paper* quotes. **"Reproduces bit-for-bit" is not true of this GPU path
and has been removed from the paper and from this file**; `index_add_` reduction order is the reason.

Reproduce the whole set with `bash scripts/rerun_sweeps_logged.sh` (sweeps, logged) and the two
per-experiment commands above; logs are in `docs/provenance/{e1,e2,determinism}_*.log`.

Consequence recorded in HANDOFF §5 Result 5: a universal constant is not supported; the ratio is
still >1 everywhere and still tracks `1/θ`, on every dataset and configuration tried.

### Section 4.7 — the gradient a reconstruction descends on (Table `tab:vjp`)

**This entry was missing, and its absence was load-bearing.** §4.7 quotes four cells of a 12-seed
sweep, but no §2 entry pointed at them: the E3 material above documents the 90-run **pre-fix**
sweep (15 seeds, `center`/`surface`), which is a different experiment kept for the failure-mode
record, and the only file §2 named in connection with `tab:vjp` was
`outputs/vjp_error_q1024_postd8.json` — a single-seed, $Q{=}1024$ spot check that is not the
table's source and does not contain these numbers. A reader following §2 to `tab:vjp` landed on
the wrong artifact family. It now points here.

| Number | Artifact |
|---|---|
| **All four cells of `tab:vjp`** (`com`/`geo` × θ=0.50/0.70, **15 seeds**, density-weighted $L$, $p{=}0$) — the manuscript's current numbers | `docs/provenance/artifacts/e3_sweep_postd8_15seed_summary_p0.json`; the 60 per-run artifacts are in `docs/provenance/e3_sweep_postd8_15seed/` and the run parameters in `e3_sweep_postd8_15seed_MANIFEST.json` |
| The $p{=}1$ companion of that run (0 of 60 runs above one) | `docs/provenance/artifacts/e3_sweep_postd8_15seed_summary_p1.json` |
| **Superseded**: the 12-seed summary the manuscript quoted until 2026-09-25 | `docs/provenance/artifacts/e3_sweep_fixed_summary_p0.json` and `..._p1.json` |
| The check that the printed table matches the artifact | `scripts/_check_vjp_table.py` |
| The check that a summary equals the summary of its own per-run logs | `scripts/_verify_e3_from_logs.py` |

**Why the table was re-measured, and what the re-measurement found.** While repairing the pointer
index it turned out that the committed 12-seed summary and the 48 per-run logs committed beside it in
the same commit (`7c56eb0`) **do not agree**: re-summarising those logs and comparing field by field,
104 of 144 values differ at the 1 % level, across all four quantities and both multipole orders. The
per-run JSONs that would adjudicate are gone, so rather than pick a winner the table was **re-measured**
on the current code with the configuration written down: `N=10^5`, `Q=256`, `max_leaf=8`, synthetic
cloud (content hash `84dcab6c…`, see §4.3), seeds 0–14, criteria `center`/`com`, θ=0.50/0.70, on CPU
with torch 2.8.0. Sixty runs, 60/60 exit 0, summarised by the repository's own `scripts/summarize_e3.py`.

The re-measurement is consistent with the old table where it should be: **every maximum reproduces to a
few per cent**, while the medians move within the 30 % platform spread that §4.7 already documents
(`com`, θ=0.50: gradient median `1.1e-4` → `1.4e-4`, maximum `1.4e-3` → `1.4e-3`). Two things changed
in the table and both are small: the amplification maximum went `0.77` → **`0.76`**
(`0.7747107442010578` → `0.757516`), and the `com`, θ=0.50 amplification median went `0.23` → `0.25`.
Every qualitative statement survives: amplification below one in all four cells, zero failures in 60
runs, `cos ≥ 0.999992`. The table is now labelled 15 seeds and back-reference-able to 60 artifacts
rather than to one summary with absent inputs.

**The disagreement is now explained, and it is the runtime stack rather than the code.** The 48 logs
were traced to the tree that produced them: `/root/autodl-tmp/Laniakea-session/` on the old rented
instance, whose `eval_vjp_error.py` is dated 2026-09-13 09:56 and whose `barnes_hut.py` is dated 12:59,
both before the logs' own 17:13. Running that tree with its own stack (Python 3.10.8,
torch 2.1.2+cu121, numpy 1.26.3) reproduces the logs **digit for digit** — checked seed by seed at
(`center`, θ=0.50) for seeds 0, 1 and 2, matching the gradient error, the per-node tidal error and the
amplification to every printed digit. Diffing that tree against this repository gives **three lines, all
of them docstring references** to a renamed document (`docs/DEFECT_D8_cell_centres.md` →
`docs/PROVENANCE.md` §3.5); `gravity.py` is byte-identical. So the 104/144 disagreement is a *stack*
difference, which is precisely the sensitivity the precision paragraph in §4.7 describes. Three stacks,
three results, and the amplification maximum each one gives:

| stack | python / torch | amplification maximum |
|---|---|---|
| local — the source of `e3_sweep_fixed_summary_p0.json` | 3.14.4 / 2.14.0+cpu | `0.7747107442010578` |
| old instance — the 48 committed logs | 3.10.8 / 2.1.2+cu121 | `0.758` |
| current — the 15-seed re-run | 3.12.3 / 2.8.0+cu128 | `0.757516` |

Two independent stacks agree at `0.7575`–`0.758`, and the local stack is 2.2 % higher — the "maxima move
by a few per cent" of §4.7. That is why the printed value is `0.76`: not a new experiment displacing an
old one, but the two mutually consistent stacks preferred over the outlier.

An error of mine is retracted here. An earlier draft of this repair concluded that the logs came from
"a code state that is no longer reachable". That was wrong: the code is identical and only the library
versions differ. `--legacy-geometry` failing to reproduce them was expected all along, because the logs
already used the fixed geometry — the flag moves the tree in the wrong direction, not the right one.

**Name mapping, which is not obvious from either side.** The manuscript's `geo`
(*geometric-centre*) row is the artifact's `center` opening mode. Same run, different label. The
manuscript's `com` is the artifact's `com`.

**Correction of 2026-09-25, and it is the kind of error this file exists to catch.** The paper
printed the `com`, θ=0.50 amplification maximum as `0.78` through revision v1.6. The artifact's
unrounded value is **`0.7747107442010578`**, which to the two significant figures the column uses
is **`0.77`**. So every other printed digit in the table reproduces exactly and that one did not:
`scripts/_check_vjp_table.py` compares all twenty printed values (4 cells × 5 quantities) against
the artifact and reports exactly one disagreement before the fix, none after. The direction
matters for how it should be read: `0.78` *overstated* the amplification, i.e. it argued against
the paper's own claim that the VJP contracts — the claim survives, the digit did not. `README.md`
had already recorded the discrepancy parenthetically ("the manuscript prints `0.78`"), which is
how it was found; documenting a mismatch is not the same as fixing it, and the lesson is that the
parenthetical should have been the trigger.

### Section 4.3 — accuracy and cost scaling

**These artifacts are pre-D8 and are no longer the paper's numbers.** They are kept because the
before/after pair is evidence for D8; the paper now quotes the post-D8 block immediately below.

| Number (pre-D8, superseded) | Artifact |
|---|---|
| forward-pass time `0.205239 → 4.155135` s, growth `20.245353×` | `outputs/scaling.json` (`scaling.hmmp_time`, `scaling.n`) |
| exact sum `0.016587` s (N=1000), `0.178275` s (N=3000) | same file (`scaling.exact_time`) |
| opening-angle force error `0.0312507 … 0.0014014` | same file, `error_vs_order.rel_err` |
| peak memory uncapped `159.85 → 11794.97` MB | `outputs/memory_hmmp_2M3_uncapped.json` |
| peak memory capped at 2.5M entries `159.85 → 3780.78` MB | `outputs/memory_hmmp_2M3_capped.json` |

`scaling.json:error_vs_order` is **misnamed**: the generating function
`measure_error_vs_order` maps its index `p` to `theta = max(0.05, 0.7/(1+0.5p))` and never changes
the multipole order, so this is an *opening-angle* sweep. The paper flags this where the numbers are
used. See §3.3.

### Section 4.3 / Appendix B — the cost tables, re-measured post-D8 on the A800 box (2026-09-17)

Run on one A800 80 GB, `CUDA_VISIBLE_DEVICES=0`, with the source verified identical to the local
checkout before the run: `md5sum` of `src/laniakea_hmmp/models/barnes_hut.py`
(`245c160b8e37ec6e8a7951f79fe5b57e`, the file containing the D8 fix `step = sz / 4.0`), `hmmp.py`,
`exact_field.py`, `scripts/benchmark_memory.py`, `scripts/scaling.py` and
`scripts/compare_d8_cost_ab.py` all match byte for byte. The box is not a git checkout, so this hash
comparison is the only evidence that it ran the same code as the manuscript.

| Number (post-D8) | Artifact |
|---|---|
| forward-pass time A800 `0.015203 → 0.726464` s, growth `47.792956×` | `outputs/scaling_postd8_a800.json` + `.log` |
| the same configuration on CPU: `0.260325 → 6.946166` s, growth `26.68×` | `outputs/scaling_postd8_cpu.json` |
| opening-angle force error `0.0032450 … 0.00018170` (θ = 0.467 … 0.175) | `outputs/scaling_postd8_a800.json`, `error_vs_order.rel_err` |
| far interactions admitted `98,272` (N=1k) → `6,978,919` (N=20k), `71.0×` | `outputs/d8_cost_ab_a800.json` |
| fixed/legacy cost ratio `2.355 / 5.388 / 10.034 / 16.373×` at N = 1k/3k/10k/20k | same file, deterministic far counts `2.578 / 3.377 / 3.938 / 4.332×` |
| peak memory uncapped `199.12 → 28,082.88` MB (N=10k → 1M); `2.3M` **OOM** under the default allocator | `outputs/memory_hmmp_2M3_uncapped_postd8.json` + `.log` |
| peak memory uncapped 2.3M with `expandable_segments:True`: `71,416.82` MB | `outputs/memory_hmmp_2M3_uncapped_expseg_postd8.json` |
| peak memory capped at 2.5M entries `198.17 → 11,351.59` MB | `outputs/memory_hmmp_2M3_capped_postd8.json` + `.log` |
| self-similar ladder `N = 10^3 → 512×10^3` (box 100→800, grid 32→256, density fixed): time `0.015864 → 15.877789` s, growth `1000.85×` vs `974.38×` for O(N log N) | `outputs/scaling_fixed_density_postd8.json` + `.log`, `scripts/scaling_fixed_density.py` |

**Two traps in this campaign, both worth recording.**

1. `outputs/scaling_legacy_geometry_box_cpu.json` was written at 20:49 under the name
   `scaling_postd8_box.json`, *before* the post-D8 source reached the box, and it reproduces the
   pre-D8 curve exactly (`rel_err` `0.0312507`, identical to `scaling.json`). It was renamed. A file
   whose name says "postd8" and whose contents are the pre-D8 geometry is exactly the kind of artifact
   that gets cited by mistake later; the tell is the `rel_err` column, not the name.
2. The gate check (`scripts/exact_nbody.py`) **did not run on the box**: it failed with
   `FileNotFoundError: data_cache/camels_2M3.npz`, because the snapshot lives at
   `/root/autodl-tmp/camels_2M3.npz` and there is no `data_cache/` directory there. The log is kept as
   `outputs/gate_exact_nbody_box_FAILED.log`. The D8 anchor for this campaign therefore rests on the
   `md5sum` comparison above and on the `d8_cost_ab` two-arm run (which prints both geometries in one
   process), **not** on the gate this checklist prescribes. Anything that depends on the gate having
   passed should be re-checked.

The header of the pre-D8 table (`0.205239 →`) also means the paper's earlier `20.2454×` growth was an
artifact of the *defective* tree, which admitted fewer far interactions and was therefore cheaper. The
geometry fix made the operator more accurate **and** more expensive: accuracy improved `9.6×` on the
opening-angle curve and cost rose up to `16.4×` at N=20k. Those two facts belong in the same sentence
and were previously in different ones.

### Section 4.3 / Appendix E — where the tree overtakes the direct sum (added 2026-09-22)

The manuscript used to say three times that it *brackets* this crossover without measuring it. It is
measurable once the brute-force sum is chunked over source rows (the harness's own `exact_force`
materialises `(N,N,3)` and so caps the comparison at `N=3000`).

| Number | Artifact |
|---|---|
| tree becomes the cheaper full-field method at `N` ~ 6,000-8,000 (log-interpolated 7,200); both runs agree on the bracket | `outputs/crossover_cpu.json`, `outputs/crossover_cpu_run2.json` + `.log` |
| sampled reference (`Q=2048`) becomes the cheaper reference at `N` ~ 1,060 (bracket 1,000-2,000) | same |
| full/sampled = `N/Q` by construction: 7.7x at 16,000, 1123x at 2.3M, 16384x at 16.7M | same |
| two runs, `tree-repeats` 5 and 7, `exact-repeats` 2 and 3; the bracket reproduces, the individual points carry ~20% noise | both files, `rows[]` |

**Hardware caveat, stated because it decides how the number may be used.** Both crossovers are CPU
measurements on a 12-thread Windows box (the A800 was retired once no GPU-only measurement remained),
and the tree benefits more from parallelism than the direct sum does, so a GPU moves both to smaller
`N`. The in-run *ratios* are robust -- both series are measured in one process, interleaved per `N` --
but the absolute level is not: the same tree forward pass at `N=1000` measured 0.233 s, 0.258 s and
0.148 s on three consecutive days of this work, i.e. the machine's own speed varies by up to 1.8x.
That is why the paper reports the bracket rather than the interpolated 7,200 alone.

### Appendix K — the particle-mesh baseline, and a boundary-condition finding (added 2026-09-22)

The paper's "is there a cheaper way?" section compared against a k-NN cutoff, which is not how anyone
computes gravity. This adds the standard cheap solver: particle--mesh, deposit -> FFT Poisson ->
finite-difference gradient -> interpolate back.

| Number | Artifact |
|---|---|
| tree (N=2.3M, Q=1024, theta=0.5): force `2.76e-3`, tidal `3.19e-2`, `31.4 s` for build + the 1024-node field | `outputs/pm_baseline_n2M3.json` |
| PM 64^3 (8.8 part/cell): force `1.98e-1`, tidal `9.60e-1`, full 2.3M-node field in `6.4 s` | same |
| PM 128^3 (1.1 part/cell): force `1.90e-1`, tidal `9.12e-1`, `8.8 s` | same |
| PM 256^3 (0.14 part/cell): force `1.80e-1`, tidal `7.84e-1`, `21.5 s` | same |
| solver control: a smooth analytic mode is reproduced to `2e-4` (`k dx`=0.10) to `1.3e-2` (0.79), with the `(k dx)^2` signature | `scripts/_diag_pm_smooth.py`, `outputs/_diag_pm_smooth.log` |
| the uniform-box (`k=0`) term is `2.3x` the peculiar field: `44.6` against `54.6` isolated and `16.1` peculiar | `scripts/_diag_pm_peculiar.py` + `.log` |
| PM's error against a cell-smoothed reference bottoms out at `53%` and never collapses | `scripts/_diag_pm_smoothing.py` + `.log` |
| earlier decomposition: PM is `0.21x` its own deposited grid's direct sum, which is what sent us looking for the convention gap | `scripts/_diag_pm_solver.py` + `.log` |

**Two things this produced besides a baseline row.**

1. **A boundary-condition finding about the operator itself.** It treats the box as isolated: no
   periodic images anywhere in `models/`. In this box the field of the mean density -- the `k=0` mode a
   periodic solve drops -- is `2.8x` the peculiar force, so the isolated convention's total is dominated
   by it. Its gradient is a constant isotropic tensor, so the *tidal* results shift by a constant trace
   and the bound is unaffected, but the *force* is not. Now disclosed in section 4.6 rather than left
   implicit. **Anyone using this operator for periodic cosmology should know this.**
2. **A convention trap that would have produced a wrong number.** Measuring periodic PM against an
   isolated reference reports `98%` error where the mesh is responsible for `18%`: `80` points of it are
   the dropped `k=0` mode. The fix is to compute the uniform-box field by summation, because
   `(4 pi G rho/3) r` is the *sphere* formula and this box is a cube, where it is `8%` wrong. Both wrong
   versions were written during this experiment and are recorded here rather than deleted.

### The isolated convention: its size, and why the force errors cannot simply be re-expressed
(added 2026-09-22)

Follow-on to the PM baseline, which is where the convention gap first showed.

| Number | Artifact |
|---|---|
| `|a_isolated|` median `48.9`, `|a_uniform|` (the `k=0` term) median `47.3`, `|a_peculiar|` median `9.1` | `outputs/isolated_convention_penalty.json` |
| the uniform term is `4.8x` the peculiar force at the median and `15.3x` at p95 | same |
| re-expressing the operator's force error on the peculiar field takes it from `2.7e-3` to a median of `4.8` (not a degradation: the two fields differ by the background) | same |
| tidal is unaffected: the background's gradient is a constant isotropic tensor and the reference tidal tensor is traceless, so the tidal error stays `3.2e-2` | same |

**Why this is a disclosure rather than a re-measurement.** The operator returns the *isolated* force and
a periodic code returns the *peculiar* one; in this box they differ by a term larger than the peculiar
force itself. So no set of existing numbers can be "converted" between the two conventions: the
per-node relative error against the peculiar field divides by a quantity the operator never computes.
A periodic version would need a norm-based force statistic -- which is what periodic codes quote, and
which would break comparability with every force row in the paper. Both the size of the gap and this
reason are now in section 4.6.

### E7 finished — the single-cell constant does not survive a geometry holdout (added 2026-09-22)

E7's question was whether `K` has a finite and stable supremum. It does (`K_tidal_max` 59.6), and the
repository has reported `eps_1 <= C rho eps_0` with `C = 5.520` tidal / `8.897` force and **0
violations in 900 samples** since 2026-09-14.

**That check could not fail.** `C = 1/min K` is taken over the same 900 samples the inequality is then
evaluated on, so it is the tangent line at the worst sample. `scripts/e7_holdout_bound.py` calibrates on
one set and evaluates on another:

| Fold | Result | Artifact |
|---|---|---|
| in-sample (reproduction check) | `C` 5.520 / 8.897 over 900 samples, binding `G7_shell` — reproduces the committed artifact exactly | `outputs/e7_holdout_bound.json` |
| leave-one-GEOMETRY-out (10 folds) | 9 of 10 covered with room to spare (`C_test/C_train` 0.02-0.43x); **`G7_shell` violates 13 of 90 held-out samples and forces `C` up 2.34x (tidal) / 4.38x (force)** | same |
| fresh-SEED holdout (stochastic families) | 0 violations in 180 unseen draws, `C_test/C_train` 0.43x / 0.23x | same |

So a universal constant exists only in the weak sense that `sup K` is finite: a `C` calibrated on
arbitrary clouds does not transfer to the near-degenerate ones, and the binding case is the spherical
shell, where the octupole is weakest. The paper reports the scaling law per dataset and the constant per
family of cell geometries. Script `scripts/e7_holdout_bound.py`, artifact
`outputs/e7_holdout_bound.json`, log beside it.

### Section 4.6 and Appendix K — the neighbour cutoff, and the conservation laws (re-measured post-D8, 2026-09-17)

The three measurements behind §4.6 and Appendix K (neighbour-cutoff baseline) and Appendix L
(conservation critic) were first taken under the pre-D8
octree geometry, and two of the paper's conclusions were artefacts of that defect. They were re-taken
after the D8 fix, on CPU (N=2,000–100,000, so no GPU was needed). Both the pre-D8 and post-D8 artifacts
are kept; the post-D8 file is the one the paper cites.

| Number | Artifact |
|---|---|
| terms per query node `664.6` (1,145,335 far + 215,667 near-pair entries over 2,048 nodes), matched `k=665` | `outputs/knn_baseline_postd8.json` |
| k-NN force rel. err `9.32e-1`, tidal `1.79e-1`; tree `p=0` `3.02e-3` / `9.97e-3`; tree `p=1` `5.34e-4` / `2.94e-3` | same file |
| momentum violation (single evaluation, N=20,000): exact `3.59e-8`, k-NN `4.55e-3`, tree `p=0` `9.20e-5`, `p=1` `9.62e-6` | same file |
| Lumyn critic channels: exact `1.69e-4 / 1.20e-6 / 1.94e-8 / 1.30e-8`; tree `p=0` `6.04e-4 / 2.33e-3 / 1.90e-4 / 9.81e-6`; `p=1` `5.12e-4 / 3.83e-4 / 1.76e-5 / 1.31e-6` (energy/angular/momentum/com) | `outputs/lumyn_critic_bridge_postd8.json` |
| over-time drift, 150 steps: exact `1.22e-4 / 4.95e-5 / 1.14e-6`, tree `p=0` `1.386 / 2.33e-4 / 4.62e-4`, `p=1` `1.139e-1 / 1.62e-4 / 6.24e-5` (momentum/energy/com) | `outputs/conservation_drift_postd8.json` |

What the D8 fix changed, and in which direction:

| Quantity | pre-D8 | post-D8 | effect |
|---|---|---|---|
| tree vs k-NN, force | `37×` | `309×` | pre-D8 understated the tree |
| tree vs k-NN, tidal | `1.6×` | `18×` | **claim flipped**: the cutoff is *not* competitive for tidal-only work |
| tree vs k-NN, momentum | `0.6×` (tree worse) | `49×` (tree better) | **claim flipped**: the tree is the better conserver, not the worse |
| `p=0` momentum drift over 150 steps | `10.4` | `1.386` | the pre-D8 violation was mostly geometry |
| angular / momentum channel ratio | `20×` | `12×` | same conclusion, smaller margin |
| energy ratio to exact (critic) | `2.5` | `3.6` | same order |

The pre-D8 counterparts are `outputs/knn_baseline.json`, `outputs/lumyn_critic_bridge.json` and
`outputs/conservation_drift.json`. Note also that the k-NN terms-per-node count *itself* changed
(`115.5 → 664.6`): D8 altered which cells are admitted, hence the operator's cost, hence the matched
`k`. Any comparison that holds `k` fixed across the defect is comparing at unequal cost. The
`outputs/vjp_error_q1024_postd8.json` file is a **single-seed, $Q{=}1024$ spot check** from §4.7 and
was not re-run here, because that section had already been re-measured post-D8 (12 seeds × 2
criteria). It is *not* the source of Table `tab:vjp` and does not contain the table's numbers —
which §2 used to imply, and used to have no alternative to; the table's actual artifacts are the
`e3_sweep_fixed_summary_p{0,1}.json` pair indexed under §4.7 above.

---

## 3. Known defects


### 3.0 Target leak in the node features (why an entire section was withdrawn)

**This is the most serious defect found, and it invalidated a whole set of results.**

`construct_cosmic_graph` built node features as

```python
rho = true_density
vel_disp = velocity.std(dim=0, keepdim=True).expand_as(velocity).norm(dim=1, keepdim=True)
x = torch.cat([torch.log1p(rho).unsqueeze(1), vel_disp], dim=1)
```

which is wrong twice over:

| # | Defect | Evidence |
|---|---|---|
| 1 | `x[:, 0]` is `log1p(true_density)`, and the training target is `log1p(g["true_density"])` — **bit-identical** | max abs difference `0.000e+00`; Pearson `r = 1.00000000`; `MSE(x[:,0], target) = 0.000000` |
| 2 | `velocity.std(dim=0)` reduces over *all* particles, so after `.expand_as()` every node received the **same** scalar | `x[:,1]` min == max == `349.1930` (std `0.000000`) |

Consequences, all measured:

* A model that copies feature 0 scores MSE `0`. The regression task had no content.
* Both features were uninformative about anything except the answer, so every method in the
  comparison sat at or near the **predict-the-mean floor**: held-out MSE `0.031807` on the 100K
  seed-0 split, against reported "wins" of `0.026257`–`0.032436`. The differences being compared
  were a few percent of a degenerate baseline.
* The leak is *exploitable*, which proves it was not harmless: with input standardisation that makes
  copying easy, ports of PolyFormer and Polynormer reached val MSE `0.0116` and `0.0026`
  respectively at 100K/15 epochs — an order of magnitude better than anything the paper reported,
  purely by reading the answer off the input.

A decisive control was run: with non-leaky kinematic features (`[|v|, v_z]`) at 100K/30 epochs.
**The claim originally recorded here was that HMMP did not win that control**, on the numbers Graph
Transformer `0.025726` against HMMP `0.027378`, with GraphSAGE `0.027686`, PolyFormer `0.027483`,
Polynormer `0.027562`, Cluster-GCN `0.031237` and a constant-mean floor of `0.031807`; the conclusion
drawn was that the reported ranking had been an artifact of the leak.

**Withdrawn: the 2026-10-19 30-epoch run was not leak-free.** Its console prompt shows it ran from
`~/autodl-tmp/Laniakea-HMMP`, whose `graph.py` still had `rho = true_density` (the leak). The
transcript and screenshot are kept under `docs/provenance/withdrawn_pre_fix/`, marked
**do-not-cite**. It is not a leak-free control and is not used here. The only leak-free 100K
30-epoch run in this ledger is the 2026-10-20 rerun, which was **not committed** (it is not cited
here); that rerun split by statistic — Graph Transformer ahead on the all-node `test_mse`, HMMP
ahead on `best_val` and `test_mse_at_best_val`.

**RESOLVED (2026-09-13).** An intermediate revision of this section declared those numbers
**unsourced**, because a search of every `outputs/*.json` in the repository found none of them. That
was wrong, and the error was ours: **the artifact existed on the GPU box and had never been
committed.** It is now in the repository at `docs/provenance/leakfree_control/probe_legit_100k.json`
and reproduces all six values exactly:

```
graph_transformer 0.02572641521692276   hmmp        0.02737845666706562
graphsage         0.02768597938120365   polyformer  0.027482518926262856
polynormer        0.027561528608202934  clustergcn  0.03123689442873001
```

The five matched `100k` seeds were fetched alongside it. That comparison was first recorded here as
making the control's verdict **N-dependent**. It is not: §3.0b shows the effect attributed to `N` is a
**node-feature artifact**, and that the leak itself was secondary to a deeper defect in the target.

| configuration | MSE winner | power-spectrum error |
|---|---|---|
| `probe_legit_100k` (leak-free probe, 100k) | **graph transformer** 0.025726 vs 0.027378 | not recorded |
| `baseline_matched_100000_seed0..4` | **mixed depending on the statistic**: graph transformer in 3 of 5 by `test_mse_at_best_val`; HMMP in 3 of 5 by the all-node `test_mse`; best 0.025557 vs 0.027315 | noisy and mixed (0.028-0.069 vs 0.007-0.091) |
| `baseline_matched_{500000,1000000}_seed0..2` | **hmmp** in 6 of 6, by 1.15-1.16x | **graph transformer 5-12x better** (0.063-0.149 vs 0.0061-0.037) |
| `2026-10-20` 100K/100ep seeds 0-2 | **mixed**: HMMP wins 2/3 on each metric; means test 0.030882 vs 0.031377, at-best 0.030020 vs 0.029230 (GT), best-val 0.029659 vs 0.028815 (GT) | HMMP 0.039959 vs GT 0.026476 |
| `2026-10-20` 500K/1M, 2 seeds each | **hmmp** on MSE: 1.11x at 500K, 1.22x at 1M vs ClusterGCN | **12x worse than GT at 500K; 466x worse than ClusterGCN at 1M**; GT OOM at 1M |

The `2026-10-19` console run is deliberately absent from this table: it used the pre-fix leaky
code and is withdrawn under `docs/provenance/withdrawn_pre_fix/`.

The complete `2026-10-20` set (12 files + README) is under
`docs/provenance/leakfree_control/2026-10-20/`: 100K 3 seeds, 500K/1M 2 seeds, the 30-episode
rerun, the leak-free arm of the A/B, and the ceiling baselines. The pre-fix arm of the A/B is
`docs/provenance/withdrawn_pre_fix/ab_leaky_100k_30ep_seed0.json` and is not leak-free.

**No recorded configuration wins both statistics.** Where the operator wins MSE it loses the power
spectrum badly; where the spectrum is close the MSE is a loss. The 30-epoch probe did not record the
power spectrum, so it cannot be used to claim a joint win. The withdrawal therefore stands, and
`paper.tex` D1b now states it in these terms rather than as a single number.

*Lesson recorded because it cost a wrong commit:* a paper number whose artifact is not in the
repository is indistinguishable from an invented number. The fix is to commit the artifact, not to
re-word the claim.

### 3.0b The leak was secondary: the target is a deterministic function of the input

The leak in §3.0 was real, avoidable, and **not the reason the section is dead**. The deeper defect is
how the target is built. `CamelsSnapshotAdapter.load()` (`src/laniakea_hmmp/data/camels.py:156`)
estimates it from the particles themselves:

```python
smooth = gaussian_filter(cic(pos, mass, grid_n=64), sigma=2.0, mode="wrap")
true_density = smooth[cell_index_of_each_particle] / mean
```

so `true_density` is constant inside each of `64³ = 262,144` cells, taking at most that many distinct
values, and is fixed by `(pos, mass)` — and `mass.std() == 0` in this dataset, so it is fixed by
**position alone**. Measured on the exact npz the campaigns trained on
(`scripts/density_target_circularity.py`):

| check | result |
|---|---|
| rebuilt target vs stored target | `max|diff| = 2.4e-07` (float32 rounding only) |
| distinct target values | **257,444** of 262,144 |
| max within-cell target std | **1.6e-07** (zero) |
| exact-cell residual over all 2.3M nodes | **2.2e-31** |
| lookup table, val nodes whose cell is in train | **8e-34 … 1.6e-31** (machine zero) |

**The Bayes-optimal MSE of this regression is exactly 0, not the `0.032` "constant-mean floor".**
Nothing that trains is needed to see the gap (`scripts/density_spatial_baseline.py`,
`scripts/density_feature_probe.py`):

| predictor at `N = 2.3M` | MSE |
|---|---|
| exact 64³ lookup | **0** |
| position-only k-NN smoother (best `k=4`) | **0.000255** |
| linear on `[|v|, local_velocity_dispersion]` | 0.020377 |
| **leak-free HMMP** (`tr_leakfree_2M3`) | **0.014661** |
| train-only 256³ cell mean | 0.021099 |
| linear on the §3.0 control's `[|v|, v_z]` | 0.028838 |
| constant-mean floor | 0.032223 |

Three consequences, each of which changes what §3.0 means:

1. **The section is withdrawn for a stronger reason than the leak.** The leak could have been fixed;
   the task cannot be made meaningful while its target is a function of its inputs.
2. **The leak was redundant.** Feature 0 of the August runs was bit-identical to the target, yet those
   runs scored `0.0248` — *worse* than the leak-free rerun's `0.014661`. With a target already
   determined by the input, the leak buys nothing, and the architecture ranking carries no information
   in either direction.
3. **The "`N`-dependent verdict" is a feature artifact.** The published 100k control used `[|v|, v_z]`;
   the A/B uses `[|v|, local_velocity_dispersion]`. At *fixed* `N = 100,000` that change alone moves
   the linear bound `0.028702 → 0.015793` (R² `0.105 → 0.511`) — a **1.8×** effect. With the features
   held fixed, `N` does nothing at all: `tr_leakfree_{100k,1M,2M3}` give `0.013049` / `0.015050` /
   `0.014661` across a 23× range in `N`, the smallest `N` being the best.

The A/B was pre-registered with the rule "~0.025 → capability genuine; 0.026–0.030 → leak-inflated".
The result fell **below both branches**, which is the outcome that exposed the framing rather than
answering it. See `POSTD8_RECONCILIATION.md` §5a–5b.

Artifacts: `circular_2M3.json`, `spatialprobe_2M3.json`, `featprobe_2M3.json`,
`tr_leakfree_{100k,1M,2M3}.json`, all in `docs/provenance/artifacts/`.

---

## 7. Post-D8 reconciliation of `tab:sampled`, and three claims it invalidated

Re-measured on the 4xA800 instance, 2026-09-13, with the post-D8 tree. Same statistic on both sides —
per-node **mean** relative error — because comparing the new `global_rel_err` against the old `mean`
would have shown a spurious 57x improvement. Same configuration otherwise: CAMELS `L25n256`,
`theta=0.5`, `epsilon=1e-4`, `max_leaf=16`, `seed=0`, `query_seed=0`.

**Every row of the table improved by `10`-`41x`:**

| p | N | Q | old force | new force | factor | old tidal | new tidal | factor |
|---|---|---|---|---|---|---|---|---|
| 0 | 10,000 | 512 | 0.027775 | 0.002834 | 9.8x | 0.124150 | 0.007310 | 17.0x |
| 0 | 100,000 | 2048 | 0.025732 | 0.001990 | 12.9x | 0.127748 | 0.008233 | 15.5x |
| 0 | 1,000,000 | 2048 | 0.025610 | 0.001716 | 14.9x | 0.139563 | 0.008306 | 16.8x |
| 0 | 2,300,000 | 1024 | 0.024929 | 0.001666 | 15.0x | 0.136239 | 0.009149 | 14.9x |
| 0 | 2,300,000 | 2048 | 0.024071 | 0.001704 | 14.1x | 0.140502 | 0.009005 | 15.6x |
| 1 | 10,000 | 512 | 0.017023 | 0.000799 | 21.3x | 0.088616 | 0.002337 | 37.9x |
| 1 | 100,000 | 2048 | 0.016752 | 0.000528 | 31.8x | 0.092640 | 0.002687 | 34.5x |
| 1 | 1,000,000 | 2048 | 0.016854 | 0.000427 | 39.5x | 0.100769 | 0.002618 | 38.5x |
| 1 | 2,300,000 | 1024 | 0.016369 | 0.000395 | 41.4x | 0.100418 | 0.002886 | 34.8x |
| 1 | 2,300,000 | 2048 | 0.016028 | 0.000424 | 37.8x | 0.100757 | 0.002976 | 33.9x |

New configuration, no prior counterpart: **16,777,216 particles**, Q=1024 —
`p=0` force mean `0.001613` / tidal `0.017358`; `p=1` force mean `0.000588` / tidal `0.009400`.

**Three claims drawn from the old table no longer hold**, and `paper.tex` now says so:

| claim | old | post-D8 |
|---|---|---|
| \(N\)-independence of accuracy | "moves by 15% either way" over 230x | force varies by **1.7x** (p=0) and **2.0x** (p=1) over **1678x**, improving with N |
| tidal/force penalty | "5-6x", flat | **2.6-5.5x** (p=0) and **2.9-7.3x** (p=1) over 1e4-2.3M, rising to **10.8x/16.0x** at 16.7M — it *grows* with N |
| quadrupole benefit | "1.4-1.5x on force, 1.4x on tidal" | **3.0-4.2x** and **3.0-3.2x** |
| force direction cosine | "degrades to 0.9996430 at 2.3M" | **0.999999-1.000000 at every N**, no trend |

The old \(N\)-independence was an artefact of D8: its error was large enough to flatten every row to
~0.025 and hide the trend. The quadrupole's value was likewise understated because D8's error swamped
the term the quadrupole removes.

**A new statistic-level result, found while checking this.** At 16.7M the quadrupole *worsens* the
**global** relative force error (`2.5e-3` vs `8.6e-4`) while *improving* the per-node mean `2.7x`
(`5.9e-4` vs `1.6e-3`) and the 90th percentile `3.6x` (`7.7e-4` vs `2.8e-3`). The global norm ratio is
dominated by a handful of outlier nodes and can invert an ordering the per-node statistics agree on.
An earlier read of this same data concluded that float32 accumulation over a ~1e6-entry interaction
list had overtaken the quadrupole's benefit; **that explanation is withdrawn** — it rested on the
global statistic alone, and the mean and p90 contradict it. This is the same failure mode as
`sec:meanisnotbound`, one statistic over.

What *is* reproducible, from `outputs/baseline_matched_*.json` and `outputs/baseline_500k_seed*.json`,
is the opposite of the original conclusion on MSE and something stronger on the power spectrum:

| paired file | model | best val MSE | test MSE | power-spectrum err |
|---|---|---|---|---|
| `baseline_matched_1000000_seed0` | hmmp | 0.027795 | 0.029369 | 0.094953 |
| `baseline_matched_1000000_seed0` | graph_transformer | 0.032178 | 0.032113 | 0.007861 |
| `baseline_matched_1000000_seed1` | hmmp | 0.028056 | 0.028022 | 0.124303 |
| `baseline_matched_1000000_seed1` | graph_transformer | 0.032457 | 0.032133 | 0.009744 |
| `baseline_matched_1000000_seed2` | hmmp | 0.027315 | 0.030079 | 0.110001 |
| `baseline_matched_1000000_seed2` | graph_transformer | 0.031780 | 0.032349 | 0.009927 |
| `baseline_matched_500000_seed0..2` | hmmp | 0.027691 / 0.027708 / 0.027850 | — | 0.149007 / 0.104870 / 0.106636 |
| `baseline_matched_500000_seed0..2` | graph_transformer | 0.032191 / 0.031941 / 0.031906 | — | 0.013096 / 0.017745 / 0.021871 |

**HMMP wins validation MSE in 9 of 10 paired runs, by `1.15`-`1.16x` across three seeds and both
`N=500`k and `N=10^6`** (and in 2 of 3 runs at `N=10^5`; in `baseline_100k_100ep_protocol.json` the
graph transformer wins marginally, `0.029368` against `0.029985`). The only run where HMMP loses badly
is `probe_gt_after_hmmp.json`, whose operator row is a broken run (best val `0.2155` against `0.027`
everywhere else).

**The reproducible reason to withdraw the comparison is the power spectrum, not the MSE**: HMMP's
power-spectrum error is `0.063`-`0.149` against the graph transformer's `0.0061`-`0.037`, i.e. five to
twelve times worse at `N >= 5x10^5`. That is consistent across every run and is the justification
`paper.tex` D1/D1b now use.

**Resolution.** The density-reconstruction section was removed from `paper.tex` entirely and the
paper was rescoped to the operator, the bound and the cost profile — none of which depends on node
features (the operator consumes only positions and masses). The feature construction is fixed in
`src/laniakea_hmmp/data/graph.py` (features are now `[|v|, local_velocity_dispersion]`, the second
computed per-node over the kNN neighbourhood) and is covered by `tests/test_graph_features.py`,
which fails if either defect returns.

**Related, and also wrong:** `mass` is a **global constant** (`0.006488`, std `0.000000`) in the
CAMELS contract npz — the particles are equal-mass. Any HMMP term that depends on mass variation is
therefore inert on this data, which is worth knowing when reading the force results.

### 3.1 `0.025891` is unattributable

> **Historical defect, retained for the record.** This was found while the paper still contained a
> density-reconstruction section, whose "capacity paragraph" compared a wide configuration's
> `0.025888` against a "hidden=128 result" of `0.025891`. That section has since been **withdrawn
> entirely** (see §3.0), so no current claim depends on either number. The defect itself — a paper
> number with no artifact behind it — is the reason this provenance file exists, and the fix it
> prompted (training scripts now record their `config`) is still in force.

The capacity paragraph of Section 5 compared the wide configuration's `0.025888` against a
"hidden=128 result" of `0.025891`. That number appears in exactly one place in the world:
`paper.tex`. Searches performed:

| Search | Scope | Hits for `0.025891` |
|---|---|---|
| local repository | all `outputs/*.json`, docs, full git history (`git log -S`) | 0 |
| GPU instance A | 59 result JSONs, all logs, `.bash_history` | 0 |
| GPU instance B | 88 result JSONs, all logs, all launch scripts, `.bash_history` | 0 |

`git log -S "0.025891"` shows it was introduced by commit `5544c7b`, which touched **only**
`paper.tex` and added no artifact.

The likely intended number is `dist_100ep.json` = **`0.026093`** — a 4-GPU, 2.3M, 100-epoch run
(`run_log.txt`), the only such run besides `dist_h256` itself and therefore the natural
counterpart. But its hidden width was never recorded, so this is an inference by elimination
rather than a record.

**Either way the capacity conclusion survives**: `0.025888` against `0.026093` is a 0.8%
difference, so doubling the width still buys almost nothing. A re-run with the config-recording
fix settles it definitively.

#### Search log

The audit was run on 2026-09-12 against every storage location that could hold an artifact. All
four locations returned zero hits; the string survives only inside a stale copy of `paper.tex`.

| # | Location | Scope searched | Hits for `0.025891` |
|---|---|---|---|
| 1 | local repository | `outputs/*.json`, `docs/`, `git log -S` across all history | 0 |
| 2 | GPU instance A, `/root/autodl-tmp` | 59 result JSONs, 25 training logs, `.bash_history`, all launch scripts | 0 |
| 3 | GPU instance B, `/root/autodl-tmp` | 88 result JSONs, all logs, `run_1M.sh`, `run_matched.sh`, `.bash_history` | 0 |
| 4 | GPU instance C, `/root/autodl-tmp` | full recursive grep | 1 — the string inside that instance's own stale `paper.tex` |

Instance connection details are deliberately not recorded anywhere in this repository, so the
remote locations above are identified by letter rather than by address.

The commands used, so the audit is repeatable:

```bash
# 1. local
git log --oneline -S "0.025891"            # -> 5544c7b, which touched only paper.tex
grep -rn '0.025891' outputs/ docs/

# 2-4. each instance
grep -rn '0.025891' /root/autodl-tmp/ 2>/dev/null
grep -o -- '--hidden [0-9]*' /root/.bash_history | sort | uniq -c
grep -rn 'dist_h256\|dist_h128' /root/.bash_history
```

Two findings from the same sweep are recorded in §1 and §3.3: the recovered wall-clock per run,
and the fact that `dist_100.log` is not the log of `dist_100ep.json` despite the similar names.

#### Fix, and its verification

A replacement run was launched on the 4-GPU instance specifically so that the number can be
recorded rather than inferred:

```bash
torchrun --nproc_per_node=4 --standalone scripts/train_real_dist.py \
  --npz /root/autodl-tmp/camels_2M3.npz --grid-n 256 --k 12 \
  --hidden 128 --layers 3 --edge-chunk 250000 --use-checkpoint \
  --max-interactions None --epochs 100 --lr 1e-3 \
  --out /root/autodl-tmp/outputs/dist_h128_100ep.json
```

This is byte-identical to the `dist_h256` invocation except for `--hidden`. Its result JSON will
carry the `config` and `config_id` fields added in this commit, which is the check that the
recording fix actually works and not merely that it compiles.


### 3.2 Launch commands are not in `.bash_history`

The `dist_*` runs were started over non-interactive SSH, and non-interactive shells do not write
`.bash_history`. Recovered commands therefore exist only for the runs started interactively,
e.g.:

```
# outputs/tr_2M3_cos.json  (1 GPU, hidden=128, cap 2.5M, cosine)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_real.py \
  --npz /root/autodl-tmp/camels_2M3.npz --seed 0 --epochs 100 --grid-n 256 \
  --hidden 128 --layers 3 --edge-chunk 100000 --use-checkpoint \
  --max-interactions 2500000 --lr-schedule cosine \
  --out /root/autodl-tmp/outputs/tr_2M3_cos.json

# outputs/dist_6card_100ep.json  (6 GPU, hidden=128, full list)
nohup torchrun --nproc_per_node=6 --standalone scripts/train_real_dist.py \
  --npz /root/autodl-tmp/camels_2M3.npz --grid-n 256 --k 12 \
  --hidden 128 --layers 3 --edge-chunk 250000 --use-checkpoint \
  --max-interactions None --epochs 100 --lr 1e-3 \
  --out /root/autodl-tmp/outputs/dist_6card_100ep.json
```

The full recovered history is in `bash_history.txt`.

### 3.3 Deliberately mislabelled file

`dist_100.log` (world=6) does **not** correspond to `dist_100ep.json` (world=4). The names are
similar but they are different runs; `dist_100ep.json`'s log is `run_log.txt`. Verified by
comparing the per-epoch validation curves.

### 3.4 The tidal tensor was not the tidal tensor (found 2026-09-12, while running the §5 order sweep)

> **This invalidates every tidal number the project produced before 2026-09-12.** The **force**
> results are unaffected: the corrected sweep reproduces the published force `ratio·θ` exactly on
> the CPU sweep, and the sampled ladder's force column at the quoted precision.

The expression

```python
contrib = 3.0 * (d.unsqueeze(-1) * d.unsqueeze(-2)) / r2.unsqueeze(-1).unsqueeze(-1)
contrib = contrib - eye * inv_r3.unsqueeze(-1).unsqueeze(-1)
```

appeared in `scripts/exact_nbody_sampled.py`, `scripts/probe_tidal_scaling.py` and
`scripts/diagnose_farfield_scaling.py`. It is not the Jacobian of `a_i = m d_i / r^3`, which is

```python
contrib = m * ( 3 d_i d_j / r^5 - delta_ij / r^3 )
```

The first term is missing a factor `1/r^3`; the two expressions agree only at `r = 1`, which is
why this survived four rounds of revision. Second defect in the same code path: **no quadrupole
contribution to the tidal tensor existed anywhere in the repository**, so the far field of every
"tidal" number was monopole-only even where the force used the quadrupole.

Evidence, both reproducible:

| Check | Result |
|---|---|
| single mass at R, `T_xx` from autograd vs the shipped expression (`scripts/diagnose_tidal_convention.py`) | R=0.2: `250.0` vs `−122.0`; R=5: `0.016` vs `2.992`; R=1: identical (`2.0`) |
| `‖legacy − correct‖ / ‖correct‖` on the n=2000 synthetic cloud | `1280.5` |
| isotropic fraction `‖tr(T)/3·I‖ / ‖T‖` | legacy `0.861`, correct `0.000` (a Newtonian tidal field is traceless) |

Consequences: the `tidal` blocks of §1a and §1c, the "tidal error ≈ 2× force error at θ=0.5"
statement, and the `paper.tex` sentence claiming a measured tidal error at 2.3M are stale. The
corrected ratio at θ=0.5 is ~2.5–3.5×. The retracted R1 "anomalous tidal θ^1.8 scaling" was
partly this defect; the corrected tidal θ-power is 3.56 against the force's 3.23.

**Resolution.** `src/laniakea_hmmp/models/exact_field.py` is now the single canonical reference
(checked against `torch.autograd` in `tests/test_tidal_tensor.py`); the operator carries both
derivatives including the quadrupole term
(`HMMPLayer.far_field_acceleration_and_tidal`, `near_leaf_acceleration_and_tidal`); the three
legacy scripts are corrected and marked. The corrected numbers are in
`outputs/multipole_order_sweep_tidal.json` (§2, Section 4.2b).

### 3.5 The octree's cell centres were offset by a full cell side (D8) — found 2026-09-13

**The defect.** `BarnesHutTree._build_tree` stepped from a parent centre by `sz/2` when the centre of
an octant of a side-`sz` cell sits at `sz/4`. Every non-root cell's recorded `cell_center` therefore
landed a full cell side from the box whose particles it describes, and the error compounds down the
tree. Verified by the octree's defining invariant — every member particle lies inside
`[centre ± size/2]` — on three point sets:

| Point set | Cells violating their own box | Worst offset (half-sides) |
|---|---|---|
| uniform random, n=2000 | 586 / 593 (**98.8%**) | **2.000** |
| Lumyn `binary_star`, n=600 | 184 / 185 (**99.5%**) | **2.000** |
| lognormal cloud, n=20000 | 4398 / 4452 (**98.8%**) | **2.000** |

A worst offset of exactly `2.000` half-sides is the signature of a factor-two error; the root cell
scored 1.00–1.09 and was correct. A second, independent defect compounded it: callers centred the
root on the positional **mean** while giving it the maximum **span** as a side, so on a non-uniform
cloud particles escaped the root box itself (measured root excess up to `1.042`). Fixing only one of
the two left 0.4–5.2% of cells violating containment; fixing both gives **0%** on all three sets.

**What it touched.** `moments()` computes the COM from particle membership and was never affected,
which is why the `com` acceptance test looked healthy throughout. But both *geometric* acceptance
tests (`center`, `surface`) measure distance to `cell_center`, so both were corrupted, and the octant
*partition* below depth 1 also used the wrong centre. The authors' own `README.md` described the test
as "the centre-of-mass criterion" while the code measured to the geometric centre.

**Consequences.** The θ=0.70 "catastrophic, intermittent collapse" reported in earlier drafts of §5,
Appendix H and E3 is **an artifact of this defect and is withdrawn**; the number of broken cells
under the pre-fix geometry was 5/15 with gradient errors reaching `∞`. Correcting the geometry:

| Measurement | pre-D8 | post-D8 | factor |
|---|---|---|---|
| exact N-body, N=2000, θ=0.5 | `0.021419` | `0.001996` | 10.7× |
| sampled N=10⁵, p=0: force / tidal | `4.09e-2` / `1.20e-1` | `3.77e-3` / `7.90e-3` | 10.9× / 15.2× |
| sampled N=10⁵, p=1: force / tidal | `1.66e-2` / `8.46e-2` | `7.36e-4` / `2.59e-3` | 22.5× / 32.7× |
| order-shift constant (21 cells) | `1.171830` | `2.374879` | — |
| E3 gradient failures at θ=0.70 | 5/15 | **0/12** | — |

**Resolution.** `BarnesHutTree` now derives a bounding-box root and uses octant centres;
`legacy_geometry=True` reproduces the old behaviour so pre-fix artifacts stay interpretable. The
invariant is asserted in `tests/test_cell_geometry.py`, which also asserts that the legacy path
still *reproduces* the defect. Every operator number re-run in this session is post-D8; artifacts are
under `outputs/d8_sampled/`, `outputs/e3_sweep_fixed/` and `outputs/*_fixed.json`.

**What is post-D8, and what is not.** Re-run since the fix, with post-fix artifacts committed:
the sampled validation (`docs/provenance/sampled_validation/exact_sampled_fixed_*`, the superseded
pre-fix files kept beside them), the per-interaction bound verification
(`docs/provenance/artifacts/bound_f64_*`), the eigenvalue metrics of Appendix I's `tab:eigen`
(whose caption reads "recomputed with the post-defect-D8 tree"), and the capped memory series at
16.7M (`docs/provenance/artifacts/memory_postd8_capped_16M7.json`).
**Not re-run since the fix: the 2.3M peak-memory series and the forward-pass-time series**, which
resolve to `outputs/memory_hmmp_2M3_*.json` and `outputs/scaling.json` — no post-D8 marker, and
absent from the released set, so their post-fix status cannot be checked there. The fix changes the
tree, so both will move. An earlier version of this paragraph also listed the E1/E2 artifacts as
not re-run; that was wrong, and `tab:eigen`'s caption is the evidence.

---

## 4. Search commands used

Reproduce the audit with:

```bash
# does a paper number exist in any artifact?
grep -rn '<value>' outputs/ docs/

# what config did a distributed run use?  (only works if the config was recorded,
# which is true only from the commit that added the `config` field onward)
python -c "import json,sys;print(json.load(open(sys.argv[1])).get('config'))" outputs/foo.json

# which run produced a given log?  compare curves, not names
grep '^epoch' <log> | head -5
```

---

## 4b. The citation checker could not see citations, so it could not report any (2026-09-13)

`scripts/_check_tex.py` extracted bibliography keys with

```python
keys = set(re.findall(r"\\bibitem\{([^}]+)\}", t))
```

but every entry in this manuscript carries an optional argument:
`\bibitem[Author(Year)]{key}`. The pattern therefore matched **nothing**. `keys` was always the empty
set, and two lines of output were degenerate in opposite directions:

| line | what it did |
|---|---|
| `missing cites` | listed **every** cited key as missing (25 of them) — a permanent false alarm, which is why it was ignored |
| `uncited bibitems` | `keys - cites` was `∅ - cites = ∅`, so it printed `NONE` **vacuously** — a check that could not fail |

Fixing it to `r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}"` made both lines real on first execution:
`missing cites: NONE`, and **seven genuinely uncited entries** that had been sitting in the
bibliography unseen —

```
kipf2017, lu2021, raissi2019, springel2005, vaswani2017, velickovic2018, zeldovich1970
```

Five were leftovers from the ICML-era machine-learning framing (GCN, GAT, FNO, PINN, transformer),
and two were references the text *should* have been citing: `springel2005` for the production-code
calibration claim and `zeldovich1970` for the eigenvalue-sign class. Both are now cited in
`paper.tex` (the Related Work section and `sec:physics` respectively), and the five leftovers are
deleted along with the rest of the ML-positioning bibliography (17 entries) when the related-work
section was rewritten for the ApJS submission.

*Lesson, and the second instance of this shape today.* A check whose failure mode is "reports
success" is worse than no check, because it also suppresses the suspicion that would have caught the
problem. The earlier instance was the same shape at a different layer: `outputs/` being gitignored
meant a quoted number could have no committed source, and the search that "found none" was correct
while the conclusion drawn from it was wrong. Both defects were in the *instrument*, not the object
measured.

---

## 4c. The AAS class was never unusable, and the note saying so was wrong (2026-09-13)

The manuscript's preamble stated, as a settled fact, that "the versions bundled with this TeX
distribution (aastex631 and aastex63) both abort at `\maketitle` ... so the class is not usable
here", and the paper was therefore kept class-agnostic. That conclusion was reached from the
*symptom* and never diagnosed. The symptom reproduces on a minimal document:

```
! Missing number, treated as zero.
<to be read again> \currCollabLimit0
l.10 \maketitle
```

The cause is a one-line gap in the class. `\currCollabLimit0` is defined **only** by
`\collaboration#1#2` (`aastex631.cls` L6962), and it is dereferenced unconditionally the first time an
`\author` is processed (L6677, inside `\doauthor`, guarded only by `\ifnum\docollabnum<1`). The
class's own fallback — the two lines that would have called `\collaboration{\the\AuthorCollaborationLimit}{}`
— are **commented out** at L1087-1088. So any document that never calls `\collaboration` aborts, which
is every document that does not use AAS collaborations.

Two working front-matter requirements, both now in `paper.tex` and both load-bearing:

1. `\collaboration{1}{}` before the first `\author`.
2. The front matter must sit **after** the document environment opens, with an explicit `\maketitle`.
   In the preamble, RevTeX's `\collaboration@sw` is still undefined and the same command fails with
   `Undefined control sequence` instead.

With those two lines, `aastex631` compiles the full manuscript cleanly: 31 pages at the time (34 now,
after the additions of the review round), 12/12 tables, no undefined references and no missing or
uncited citations. The only warning is the class's own benign `Label 'FirstPage' multiply defined`.

> **Correction (added 2026-09-13).** This subsection originally also claimed "correct AASTeX631 front
> matter". **That was wrong**, and §4e records why: the front matter was emitted twice, because the
> `\maketitle` added alongside the `\collaboration` fix causes a second title block. The build here
> verified that the manuscript compiles and that the front matter exists — not that it appears once.

*Why this is recorded.* "The tool is broken" is a conclusion that ends work, and it was drawn here
without evidence beyond an error message. It cost the paper its submission format for as long as it
stood. The three defects recorded in this section — the citation checker that could not see
citations (§4b), the gitignored `outputs/` that made a number unverifiable (§3.0), and this — are all
the same failure: an instrument was trusted at the layer where it had stopped working.

---

## 4d. Every literature candidate behind D7 is read, and none is a collision (2026-09-13)

Defect D7 (Appendix `app:provenance`) scoped the novelty claim around one unread line. Six documents
were unreached — five left by an automated sweep that hit paywalls and CAPTCHAs, plus the classical
Cheng–Greengard–Rokhlin paper that the D7 text itself named as the most likely place a derivative
bound already existed. **All six are now read**, from **versions of record** where a DOI could be
confirmed. This section is the public receipt, and it is self-contained: the per-item reading notes
live in the working tree, but every count quoted below is reproducible from the documents themselves.

The criterion, applied to each: does the work state
`‖ second- (or higher-) derivative error of the truncated operator ‖ ≤ C(order, separation ratio, cell
geometry) × (moment norm) / r^power`? Both a **derivative** bound and an **explicit order
dependence** must hold.

| # | Work | Route / identifier | Bound? | Derivative? | Order dependence? | Verdict |
|---|---|---|---|---|---|---|
| 1 | CloudNFMM (neural operator inspired by FMM) | OpenReview `AZUkoHjSEx`, ICLR 2026 submission | no | — | — | not a collision |
| 2 | Optimizing Translation Operator Calculation in MLFMA Based on Neural Network Integration | `10.1109/IMWS-AMP62793.2024.10966730` (IEEE 10966730) | no | no | only as a network input feature | not a collision |
| 3 | A Fast Computing Technology of Three-dimensional MLFMA Based on Deep Learning | `10.1109/PIERS62282.2024.10618348` | no | no | only as cost, `O(N^{1.5})` | not a collision |
| 4 | A Mathematical Guide to Operator Learning | arXiv:2312.14688 | yes | no | Fourier-mode / trunk width, not expansion order | not a collision |
| 5 | Neural Shape Operator Surrogates — Expression Rate Bounds | Harbrecht & Schwab, preprint | yes, a priori | parametric, not on a remainder | parameter count `N`, not order | not a collision |
| 6 | Cheng, Greengard & Rokhlin, A fast adaptive multipole algorithm in three dimensions | `10.1006/jcph.1999.6355`, *JCP* 155(2), 468–498 | yes, on Φ | **no** | yes, but on the potential | **not a collision — closes D7** |

**Item 2 is the one that mattered.** It is a *learned* multipole translation operator — the grey zone,
the only place a truncation bound on learned hierarchical machinery could plausibly have been hiding.
It contains no error analysis at all: `bound` occurs zero times in three pages, `truncat` occurs three
times and every one is the definition of the expansion order or that order being fed to the network as
an input, and the sole `theorem` is the **addition theorem** for the scalar Green's function, which is
the identity the method is built on rather than a statement about its error. Its evidence for the
approximation is "the two curves are almost identical".

Item 5 is the strongest of the five and is filed as adjacent, not as a collision: it does bound the
error of a truncated spectral expansion with explicit dependence on the retained index, and it does
bound parametric derivatives of the solution. Neither half is about the second-derivative remainder of
a truncated hierarchical interaction operator — it never uses the word `multipole` in 28 pages, and
its error axis is the number of parameters, not the expansion order. It joins Neural-HSS, MSNN and
item 4 as an instance of the a priori operator-learning pole the claim already brackets.

**Item 6 — Cheng, Greengard & Rokhlin (1999) — READ, AND IT CLOSES D7.** This was the one item the
five above did not cover, and the only one the paper named as a live gap: `10.1006/jcph.1999.6355`,
*Journal of Computational Physics* **155**(2), 468–498, 31 pages, obtained by the author. It is the
classical FMM error analysis, and therefore the most plausible place a pre-existing **derivative**
remainder could hide. **It does not contain one.**

What it does contain, read from the paper itself:

| result | bounds |
|---|---|
| Theorem 2.1 (multipole expansion), eq. (7) | the potential Φ |
| Theorem 2.2 (local expansion), eq. (10) | the potential Φ |
| Theorem 2.3 (translation of a multipole expansion), eq. (15) | the potential Φ |
| Theorem 2.4 (multipole→local conversion), eq. (18) | the potential Φ |
| Theorem 2.5 (translation of a local expansion) | the potential Φ |
| Theorems 2.6, 2.7 (rotations) | **exact identities — no remainder at all** |
| the numerical error column of Tables I–XIII | a measured L₂ error in Φ, eq. (57) |

Exhaustive counts over all 31 pages, extraction validated first (both engines, ~67.8k characters, no
page yielding less than 967 characters): `derivative` **0**, `gradient` **0**, `Hessian` **0**,
`second derivative` **0**, `error bound` **0**, `error analysis` **0**, `remainder` **0**,
`convergence` **0**. All 14 occurrences of `error` are inside the timing tables; all 4 of `bound` are
the word *boundary*; the single occurrence of `force` is Barnes & Hut's title in the reference list.

**Consequence for D7.** The paper no longer has an unread literature gap it points at. D7's wording,
and the Introduction, §2, the novelty statement and the bibliography entry for `cheng1999fast`, now
say that the strongest candidate was **read** and bounds the potential only — a finding, not a hope.
What remains unread is the literature that *follows* that line, and the claim is scoped accordingly.

*Why this is recorded.* D7 was the one place where the paper asked a reader to take a limitation on
trust: "we could not obtain it". A library is not a research method, and an access failure reads
identically whether the paper contains a collision or nothing at all. Reading it converted the
paper's largest residual risk into a positive statement, and it cost one document.

*Why this is recorded.* The five-item list existed because the earlier sweep failed on access, and an
access failure is indistinguishable from a negative result unless someone goes back. One of the six
was retrieved by the author on a campus network; four were supplied as PDFs; this one was supplied as
a PDF after the review round. Every count above was taken from an extraction that was validated for
quality first — per-page character counts, a page-1 word count, and a printed prose sample — because a
gluing or glyph-dropping extractor produces a zero that looks exactly like a clean answer. That is the
same failure mode as §4b.

---

## 4e. The title block was printed twice, and the fix that caused it was ours (2026-09-13)

A reviewer reported that page 2 carried front-matter residue. It does, and it is worse than residue:
**the draft stamp, the title, the affiliation and the keywords were each emitted twice**, the abstract
sat alone on page 1, and the duplicate block had lost the author line. Verified on the exact PDF under
review (`paper.pdf`, 377,964 bytes): title marker ×2, `Keywords:` ×2.

The cause is an explicit `\maketitle`. AASTeX 6.3.1 emits the front matter itself, and **the
[AASTeX v6.3.1 Author Guide](https://journals.aas.org/aastex-v6-3-1-author-guide/) never mentions
`\maketitle` at all**. The command was added during an earlier round of this same session as part of a
fix for a different problem — the `\currCollabLimit0` abort — and it introduced a second defect while
repairing the first.

Established by building each candidate rather than by reading the class source:

| variant | title emitted | keywords emitted | result |
|---|---|---|---|
| abstract, keywords, `\maketitle` (was) | 2 | 2 | duplicate |
| abstract, `\maketitle`, keywords | 2 | 1 | duplicate |
| keywords, abstract, `\maketitle` | 2 | 2 | duplicate |
| no abstract and no keywords, with `\maketitle` | 2 | — | duplicate |
| `[twocolumn]`, no abstract, with `\maketitle` | 2 | — | duplicate |
| **no `\maketitle`** | **1** | **1** | **correct** |

Every ordering that compiles duplicates, and so does a document with neither an abstract nor keywords,
under either class option — so no reordering fixes it. Removing `\maketitle` leaves one complete front
matter block *and restores the author line the duplicate had dropped*.

*The lesson is about the verification, not the command.* The earlier round checked that the manuscript
**compiles** and that the front matter **exists**; it never checked that the front matter appears
**once**. A build that exits zero and a front matter that is present are exactly the two properties a
duplicated title block also has.

**Also fixed in the same review round:** the abstract was 583 source tokens and **290 rendered words**
against the AAS limit of 250 for ApJ/ApJS/AJ/PSJ — the Guide states that submissions over the limit
"will be asked to reduce the abstract word count". It is now 248 rendered words, measured from the
extracted text of the built PDF between the `ABSTRACT` and `Keywords:` lines rather than from the
source, because LaTeX markup inflates the source count by roughly 45%.

---

## 5. Second-derivative error law and self-certification (2026-09-13)

New in this session. Every number below comes from a script in `scripts/` and is reproducible on CPU
with `.venv`.

| Claim | Artifact | Script |
|---|---|---|
| Exact constants `mu_l = (2l+1)sqrt(l+1)`, `Lambda_l = (2l+1)sqrt((l+1)(l+2)(2l+3))`, verified to `2.8e-7` for `l <= 12` against fourth-order finite differences | `outputs/hessian_constant.log` | `scripts/_hessian_constant.py` |
| Per-interaction second-derivative bound: **no violation** over 3 aspects x 2 orders x 10 radii x 32 directions x 3 seeds; looseness `11.5x` (cube, `l<=0`) to `31.8x` (slab 1/8, `l<=2`) with exact harmonic norms, versus `98x`-`5479x` with the extent-only `M a^l` stand-in | `outputs/error_law_bound.json`, `.log` | `scripts/error_law_bound.py` |
| The first version of the constant, `sqrt(3/2)(l+1)(l+2)`, is the zonal-only maximum and sits `0.60`-`0.74x` **below** the measured remainder, i.e. was never a bound. Retained as `bound_hess_zonal_invalid` and reported in the paper in a column labelled as not a bound | same artifact, `[invalid]` column | same |
| Order-refinement surplus is the error: `abs(err)/abs(surplus)` median `1.008` (range `1.000`-`1.803`), sign alignment median `0.998`, Richardson extrapolation gain median `1.24x` (min `0.68x`) | `outputs/selfcertify_surplus.json`, `.log` | `scripts/selfcertify_surplus.py` |
| Magnitude (unsigned) surplus ratio anti-corrects: gain came out `0.76x` **purely** from the missing sign. Kept on the record in the script docstring and in paper `sec:selfcertify` | `.log` of the first run | `scripts/selfcertify_surplus.py` |
| Angle-refinement surplus on the **shipped operator does not work**: reported bar underestimates the true error by `1.57x`-`9.42x` (median `3.43x`), direction alignment `~0.0`, extrapolation gain `0.97x` | `outputs/selfcertify_operator.json`, `.log` | `scripts/selfcertify_operator.py` |
| Principal-axis direction cosine `0.9967` mean against `0.792` worst, and the relative shift of `lambda_max` `3e-2` mean against `3.6e-1` worst, all at θ=0.90 on a synthetic lognormal cloud, `N=20,000`, `Q=1024` | `docs/provenance/artifacts/r1_hessian_error_structure.json` | `scripts/r1_hessian_error_structure.py` |

**Manuscript changes made in this session**, all recorded in the table below:

| ID | Defect | Status |
|---|---|---|
| F1 | The opening criterion was stated backwards in two definitional places (`opened iff R/r<theta`), contradicting both `barnes_hut.py:427` (`# Far field: keep cells that open`) and three correct uses in the same paper | FIXED |
| F2 | The A3 summary constant `C_p <= (2p+5)/(1-theta)` is not an upper bound: the series coefficients grow. Correct form adds `2theta/(1-theta)^2`; `14` not `10` at `theta=0.5, p=0` | FIXED |
| F3 | A bound's constant was compared against a measured aggregate, and the `(p+1)(p+2)` structure was rejected although it is the correct one (sharp ratio `0.6`) | FIXED |
| F4 | `2.98/3.39` and `3.23/3.56` quoted for the same quantity with no normalisation label; `theta_power_abs_*` vs `theta_power_rel_*` | FIXED (labels); `classical 3.0` provenance still OPEN |
| F5 | Appendix `app:diff` still reported the pre-D8 `theta=0.70` collapse (max amplification `0.714`, infinite gradient error) while main text `sec:vjp` reported the post-D8 numbers (max `0.775`, zero failures) | FIXED |
| F6 | The first draft of the new `app:secondorder` constant was not a bound (see above) | FIXED |
| — | The manuscript understated JZ-FMM's validation scale (`N~10^5`); the paper itself reports direct-summation validation at `N=1024^2` and at `4e7` | FIXED |

**Open items carried forward.** Two are unresolved. They were previously held in a separate errata
working note that is not part of the released repository, so they are recorded here — this is the
shipped copy and it must be complete on its own.

- **`classical 3.0` has no provenance.** The $\theta$-power reference value $3.0$ appears in section 1
  of this file, in the row giving the measured powers for the force and the tidal tensor, where it is
  labelled "(classical `3.0`)". It has no derivation and no citation. It no longer appears in the
  paper: `paper.tex` §sec:order reports only the measured powers ($2.98$/$3.39$ absolute,
  $3.23$/$3.56$ relative) and explicitly says neither pair is the bound's own $\theta^{p+2}$, so what
  remains is a repository-level provenance gap rather than a manuscript one. Either a source must be
  produced or the label dropped.
- **A3 writes `R_c` where the second-order appendix writes `a`.** A3 bounds the remainder with the
  cell's geometric size $R_c$ and bounds the moments by $|y_a|\le R_c$, but the expansion is about the
  **centre of mass**, so the length that actually enters the remainder is $a=\max_b|y_b|$ — the
  distance from the centre of mass to the farthest particle. Appendix `app:secondorder` states this
  explicitly: the acceptance test bounds $R/r$, the bound needs $t=a/r$, and the two differ by the
  dilation $(a/R)^{l_{\min}}$ — measured $0.837$ for a uniform cube, $0.698$ and $0.693$ for slabs of
  aspect $1/4$ and $1/8$, and $\sqrt3=1.732$ for a cell whose mass sits in one corner. A3 itself
  still writes $R_c$ without the caveat. **Low risk, and not an error:** $|y_a|\le R_c$ remains valid,
  it is merely loose. But two appendices of one paper should use one notation.

**Not yet re-measured.** The D8 fix still invalidates every CAMELS-derived number
(`docs/provenance/sampled_validation/*`, E1/E2 artifacts, memory and time tables); that re-measurement
needs the GPU box and is unchanged by this session. The new law and the surplus measurements are all
synthetic or single-cell and are labelled as such in the manuscript.

---

## 6. Self-certification on the real tree: the negative result (2026-09-13, round 2)

| Claim | Artifact | Script |
|---|---|---|
| Order-general far field (any even `l`) reproduces the shipped operator: pure-far at `l=0` vs a longhand monopole sum to `1.7e-16`; pure-near vs the shipped near-only to `7.3e-07`; complete at `l=2` vs shipped to `5.5e-05`/`6.9e-05` (the float32 limit of the shipped path) | `scripts/_diag_any_order.py` | same |
| **Defect: the far field was exactly `5.0x` too small.** `_cell_rows` expanded each UNIQUE cell once and mapped rows back through `inv`, which can credit only one of the query nodes sharing that cell, so a cell admitted by N nodes contributed to one of them. A constant factor in the norm is the signature | `src/laniakea_hmmp/models/multipole_any_order.py` (`_cell_rows` docstring) | `scripts/_diag_any_order.py` |
| **The surplus does NOT survive assembly.** Real tree, one fixed interaction list, orders `{0,2,4,6}`, two seeds, 64 query nodes: per-node `abs(err)/abs(surplus)` median `0.27` and `0.15` with ranges spanning 8-10 decades; per-node `-cos(err, surplus)` median `-0.12` and `-0.25`; per-node bar calibration median `0.71` and `0.34` with a worst case of `3.5e7` | `outputs/selfcertify_order_operator.json`, `.log` | `scripts/selfcertify_order_operator.py` |
| Angle refinement fails the same way (`1.57x`-`9.42x` understatement) so neither family is shippable | `outputs/selfcertify_operator.json`, `.log` | `scripts/selfcertify_operator.py` |
| Diagnostics landmine removed: a mass-accounting check that counted QUERY particles instead of SOURCE particles reported nonsense (`61/1200` covered once). Deleted rather than fixed, because the decisive check is the cross-check and the lists were confirmed identical | `.log` of the first run | `scripts/selfcertify_order_operator.py` |

**Interpretation, and why it is the paper's argument rather than a setback.** The assembled far-field
error is a sum over many admitted cells with different convergence ratios and largely uncorrelated
error directions, so one surplus vector cannot represent it and an unlucky borderline cell produces a
large true error the surplus does not flag. A bar that understates is worse than no bar because it
certifies what it should have caught. This is "a mean is not a bound" one level up: **an error
estimator is not a bound either.** The paper's certification claim therefore rests on the provable
per-interaction bound plus the external `O(QN)` reference, not on a self-reported bar.

### 6b. The row above is not traceable, and the sweep that replaces it (2026-09-18)

**The artifact named in the row above, `outputs/selfcertify_order_operator.json`, is not in this
repository and never was.** `git log --all --diff-filter=A -- '*selfcertify*'` returns the three
scripts and `src/laniakea_hmmp/models/selfcertify.py` and nothing else, because `outputs/` is
gitignored and that file was never force-added; it is also absent from disk. So the numbers in that
row --- the `0.27`/`0.15` medians, the "8-10 decades" range, the `0.71`/`0.34` bar calibrations and the
`3.5e7` worst case --- are **unattributable**, and so are the single-cell numbers above them
(`outputs/selfcertify_surplus.json`) and the angle family's row (`outputs/selfcertify_operator.json`).
That is a second exception to the traceability claim, beside the far-field-only pair; the README had
named only one.

Re-run on 2026-09-18 with the shipped script, `Q=256`, `com`, `max_leaf=16`, `max_depth=16`, orders
`{0,2,4,6}`, over **four tree sizes and 30 seed-runs (7680 sampled nodes)**. All of it is committed:
`outputs/selfcertify_order_n*_seed*.json` + `.log`, plus
`outputs/selfcertify_order_n2000_seedsweep.{json,log}` and the summary
`outputs/selfcertify_order_across_N.json`.

| `N` | seed-runs | bar calibration: median of the per-seed medians | worst per-seed median | worst node anywhere | `abs(err)/abs(S)` median | alignment median |
|---|---|---|---|---|---|---|
| 2,000 | 18 | 0.394 | 3.05 | 8.5e5 | 0.17-0.24 | -0.21 .. -0.01 |
| 8,000 | 4 | 0.654 | 3.15 | 5.4e4 | 0.22-0.30 | -0.15 .. -0.05 |
| 20,000 | 4 | 1.458 | 15.3 | 8.3e5 | 0.36-0.39 | -0.10 .. -0.07 |
| 100,000 | 4 | 5.227 | 17.1 | 4.6e3 | 1.54-1.98 | -0.06 .. -0.01 |
| **all 30** | | **0.591** | **17.1** | **8.5e5** | 0.17-1.98 | -0.21 .. -0.01 |

**What this establishes --- and one thing it corrects in the first draft of this note.**

1. **The failure is a property of the method, not of one tree.** It reproduces at every size from
   `2e3` to `1e5` and in 30 of 30 seed-runs: the per-node understatement always develops a tail.
2. **But the bar is roughly *honest at the median*.** The median-of-medians calibration is `0.591`,
   i.e. at the typical node the surplus bar errs conservative by about `1.7x`, and only 9 of the 30
   runs have a median above 1. That is consistent with the `0.71`/`0.34` medians in the row above ---
   so those two numbers were never the problem. The failure is entirely in the tail.
3. **And the tail is what kills it.** The largest per-node calibration anywhere is `8.5e5`, and only
   2 of 30 runs understate at *every* node --- so an operator reporting its median would look fine and
   be catastrophically wrong at the nodes that matter. That asymmetry is the finding, not the median.
4. **The alignment carries no sign information**: median `-cos(err, surplus)` lies between `-0.21` and
   `-0.01` in every run, against `0.998` on the single isolated cell. The surplus is an exact
   estimator for one interaction and nothing survives assembly.

**On the `3.5e7`.** This sweep does not reproduce it: 30 runs, worst node `8.5e5`. But that is the
same tail and only `~40x` beyond what we now measure, so the value is *plausible as a different draw*
rather than contradicted. Our handling: the paper quotes the reproducible sweep, and `3.5e7` is kept in
the record as "reported once, not reproducible here". If the original run is ever recovered, rewrite
this paragraph rather than deleting it.
