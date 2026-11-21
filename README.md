# Laniakea: a differentiable Barnes-Hut operator, with a second-derivative error bound

**Minimal reproduction package.** Everything here exists so a reader can check the
paper's *operator-level* numbers on a CPU, from a clean clone.

The full development repository (CAMELS adapters, graph-network baselines, the
training pipeline, and every negative result) is kept private. This repository is the
subset the paper's claims actually depend on — and where a claim is **not** checkable
here, this README says so instead of implying otherwise.

## What is claimed, and how to check it

| Claim in the paper | Command |
|---|---|
| Complete-operator relative force error **0.001996**, direction cosine **0.999998** | `python -m pytest tests/test_farfield_correctness.py -q` |
| Octree cell-centre defect (sub-cell stride `s/4`, not `s/2`) | `python -m pytest tests/test_cell_geometry.py -q` |
| Monopole→quadrupole error ratio × θ is constant in θ, over 21 settings | `python scripts/sweep_multipole_order.py` |
| Tidal (second-derivative) product, and where it stops being measurable | `python scripts/sweep_multipole_order_tidal.py` |
| The operator's own test suite (tree build, near/far split, multipole moments, tidal tensor vs autograd) | `python -m pytest tests -q` |

Every number quoted in the manuscript is traced to a script and an output file in
[`docs/PROVENANCE.md`](docs/PROVENANCE.md). That file is the ledger, and it also lists
the numbers that were **withdrawn** and why — it is deliberately not a highlight reel.

## Install

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

CPU is enough; nothing in this repository needs a GPU.

## Run

```bash
python -m pytest tests -q                 # operator test suite
python scripts/sweep_multipole_order.py   # order-dependence table
```

Timing numbers are hardware-dependent and are **not** asserted anywhere, so a slow
machine will not fail a test.

## What this package deliberately does not contain

- **CAMELS ingestion** (`h5py`, `hdf5plugin`) — the CAMELS checks need simulation
  snapshots that are not redistributable here.
- **The radius/kNN graph builder and the learned baselines** (`torch_geometric`, `scipy`) —
  Graph Transformer / GraphSAGE / Cluster-GCN are **implemented in the private repository but
  are not used for any result in the paper** (the manuscript compares against physics solvers:
  brute-force exact N-body, neighbour-cutoff, particle-mesh). They are excluded here because
  requiring `torch_geometric` would make a clean install fragile, not because they were
  withdrawn.
- **The training pipeline and the loss functions** (the topological term needs GUDHI).

Consequently there is no timing benchmark in this package: the far-field timing table
in the paper was produced by a script that depends on the graph builder above.

## Paper source

The LaTeX source of the preprint lives in [`paper/`](paper/): `paper/main.tex` (plain
`article` build) and `paper/appendices.tex`, with figures in `paper/figures/`. Build it
with a stock TeX Live installation — no custom class file is needed:

```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Author

**Yuchen Wang** — Department of Computer Science and Technology, Tsinghua University,
Beijing 100084, China · `yagamilight0620@qq.com`  
**Xin Wang** — Xiamen Xiangyu Group Co., Ltd., Xiamen 361000, Fujian, China

## Citation

If this package is useful to you, please cite the preprint:

```bibtex
@misc{wangwang2026barneshut,
  title  = {Differentiable Barnes--Hut Gravity: A Provable Second-Derivative Error
            Bound, and Its Validation at 16.7 Million Particles},
  author = {Wang, Yuchen and Wang, Xin},
  year   = {2026},
  note   = {Preprint, arXiv identifier to be assigned}
}
```

The identifier is filled in once the preprint is announced; until then the citation
above is deliberately marked as pending rather than pointing at a dead link.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
