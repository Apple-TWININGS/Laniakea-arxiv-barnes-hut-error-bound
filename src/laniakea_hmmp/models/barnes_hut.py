"""Differentiable Barnes-Hut octree and multipole moments.

Core of HMMP. The tree *structure* is a pure function of the fixed input positions,
so it is built once without autograd. Differentiability flows through the
**multipole moments**, which are weighted sums of node masses (learnable / node
features) aggregated bottom-up over the tree. That lets gradients back-propagate
through the far-field approximation, which is exactly what makes HMMP a learned,
differentiable surrogate of the O(N^2) force summation.

Two performance-critical pieces live here:
  * ``moments`` -- vectorised bottom-up multipole aggregation (a few depth-level
    scatter-adds, not a Python loop over every cell).
  * ``build_interaction_list`` -- a vectorised Barnes-Hut descent that produces a
    **sparse** list of ``(node, cell)`` far-field interactions (O(N log N)), instead
    of a dense ``(N, K)`` matrix. This is what lets HMMP scale to millions of nodes.

Classical error analysis reference:
  Barnes, J. & Hut, P. (1986), "A hierarchical O(N log N) force-calculation
  algorithm", Nature 324, 446-449.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["BarnesHutTree", "multipole_moments_from_tree", "exact_force"]


def _morton_interleave(points: np.ndarray, bits: int = 12) -> np.ndarray:
    """Morton (Z-order) code for points in [0,1)^3, bit-interleaved into int64."""
    max_code = (1 << bits) - 1
    xi = np.clip(points[:, 0], 0.0, 1.0 - 1e-12) * (max_code + 1)
    yi = np.clip(points[:, 1], 0.0, 1.0 - 1e-12) * (max_code + 1)
    zi = np.clip(points[:, 2], 0.0, 1.0 - 1e-12) * (max_code + 1)

    def interleave(vals: np.ndarray) -> np.ndarray:
        v = vals.astype(np.int64)
        out = np.zeros(v.shape[0], dtype=np.int64)
        for b in range(bits):
            out |= ((v >> b) & 1) << (3 * b)
        return out

    return interleave(xi) | (interleave(yi) << 1) | (interleave(zi) << 2)


class BarnesHutTree:
    """Octree over particle positions.

    Static (numpy int/float) attributes
    -----------------------------------
    pos          : (N, 3)  particle positions
    cell_center  : (K, 3)  cell geometric centres
    cell_size    : (K,)    cell side length
    cell_depth   : (K,)    depth (root = 0)
    cell_parent  : (K,)    parent index (-1 for root)
    cell_children: (K, 8)  child indices (-1 if empty)
    cell_is_leaf : (K,)    bool
    leaf_of      : (N,)    leaf-cell index per particle

    Differentiable moments, via :meth:`moments`
    -------------------------------------
    com         : (K, 3)     centre of mass
    total_mass  : (K,)       monopole moment
    quadrupole  : (K, 3, 3)  quadrupole moment about the cell COM
    """

    def __init__(
        self,
        positions: torch.Tensor,
        center: torch.Tensor | None = None,
        size: float | None = None,
        max_leaf: int = 16,
        max_depth: int = 24,
        bits: int = 12,
        pad: float = 1.0,
        legacy_geometry: bool = False,
    ):
        """Build the octree.

        The root box is derived from the particle set, not from the caller. Two defects (D8a/D8b,
        see ``docs/PROVENANCE.md``, section 3.5) made the historical construction geometrically
        inconsistent: child centres were offset by ``sz/2`` instead of ``sz/4`` (D8a), and callers
        centred the root on the positional *mean* while sizing it by the max *span*, so particles
        fell outside the root box (D8b). Together they put ~99% of cells' recorded
        ``cell_center`` outside the box whose particles they describe, which corrupted both the
        ``"center"`` and ``"surface"`` opening tests.

        Now:

          * the root is centred on the **bounding-box centre**, which is required for octants to
            tile the domain, and its side is ``max(bbox_span, size) * pad`` so every particle is
            inside it by construction; and
          * child centres sit at the octant centres, ``parent_centre +/- sz/4``.

        Args:
            positions: ``(N, 3)`` particle positions.
            center: retained for backwards compatibility; ignored unless ``legacy_geometry``.
                The root centre is always the bounding-box centre now.
            size: lower bound on the root side, before ``pad``. ``None`` means "just the bbox".
            pad: multiplier on the root side. ``size_pad``-style padding belongs here.
            legacy_geometry: reproduce the pre-fix behaviour exactly (caller-supplied centre and
                size, child offset ``sz/2``). Only for re-deriving already-committed artifacts;
                expects both ``center`` and ``size``.
        """
        pos_np = positions.detach().cpu().numpy().astype(np.float64)
        self.pos = pos_np
        self.n = pos_np.shape[0]
        self.max_leaf = max_leaf
        self.max_depth = max_depth
        self.bits = bits
        self.pad = float(pad)
        self.legacy_geometry = bool(legacy_geometry)

        if self.n == 0:
            raise ValueError("cannot build a tree over zero particles")
        lo = pos_np.min(axis=0)
        hi = pos_np.max(axis=0)
        bbox_center = torch.from_numpy((lo + hi) / 2.0)
        bbox_span = float(np.max(hi - lo))

        if self.legacy_geometry:
            # Reproduce the pre-fix construction. ``center`` reproduces defect D8b as well (the
            # mean-centred root); omit it to get only the D8a child-offset defect on a
            # bounding-box root. ``size`` likewise defaults to the bbox span.
            self.center = center.detach().cpu() if center is not None else bbox_center
            self.size = float(size) if size is not None else bbox_span
        else:
            self.center = bbox_center
            self.size = max(bbox_span, float(size) if size is not None else 0.0) * self.pad

        # Normalise to [0,1)^3 for Morton ordering. Clipped, so a particle on the root face is
        # still ordered consistently.
        lo = self.center.numpy() - self.size / 2.0
        norm = (pos_np - lo) / self.size
        morton = _morton_interleave(norm, bits=self.bits)
        order = np.argsort(morton, kind="stable")
        self.order = order

        self._build_tree(order, morton)

    # -- construction -----------------------------------------------------
    def _build_tree(self, order: np.ndarray, morton: np.ndarray) -> None:
        """Iteratively subdivide along the Morton-sorted particle sequence."""
        centers: list[np.ndarray] = []
        sizes: list[float] = []
        depths: list[int] = []
        parents: list[int] = []
        children: list[np.ndarray] = []
        is_leaf: list[bool] = []

        def add_node(ctr, sz, depth, parent) -> int:
            idx = len(centers)
            centers.append(ctr)
            sizes.append(sz)
            depths.append(depth)
            parents.append(parent)
            children.append(np.full(8, -1, dtype=np.int64))
            is_leaf.append(False)
            return idx

        root = add_node(self.center.numpy().copy(), self.size, 0, -1)
        stack = [(order, 0, self.center.numpy().copy(), self.size, root)]
        node_particles: dict[int, np.ndarray] = {root: order}

        while stack:
            sel, depth, ctr, sz, node = stack.pop()
            if depth >= self.max_depth or len(sel) <= self.max_leaf:
                is_leaf[node] = True
                continue

            half = sz / 2.0
            octants = np.zeros(len(sel), dtype=np.int64)
            for d in range(3):
                octants |= (self.pos[sel, d] >= ctr[d]).astype(np.int64) << d

            for o in range(8):
                mask = octants == o
                csel = sel[mask]
                if csel.size == 0:
                    continue
                # The centre of an octant of a cell of side ``sz`` sits at
                # ``parent_centre +/- sz/4``. The historical (defect D8a) offset was
                # ``+/- sz/2``, i.e. twice that, which put the recorded centre a full cell-side
                # away from the box the cell actually occupies -- ~99% of cells had member
                # particles outside [centre +/- size/2], with a worst case of exactly 2.0
                # half-sides (``scripts/_diag_cell_geometry.py``). Both the "center" and
                # "surface" opening tests measure distance to that point, so the defect
                # propagated into them; ``moments`` (and hence the COM) is computed from the
                # particles and was unaffected. ``legacy_geometry`` restores the defect.
                step = half if self.legacy_geometry else sz / 4.0
                cctr = ctr.copy()
                for d in range(3):
                    cctr[d] += step if ((o >> d) & 1) else -step
                cidx = add_node(cctr, half, depth + 1, node)
                children[node][o] = cidx
                node_particles[cidx] = csel
                stack.append((csel, depth + 1, cctr, half, cidx))

        self.cell_center = np.asarray(centers, dtype=np.float32)
        self.cell_size = np.asarray(sizes, dtype=np.float64)
        self.cell_depth = np.asarray(depths, dtype=np.int64)
        self.cell_parent = np.asarray(parents, dtype=np.int64)
        self.cell_children = np.stack(children, axis=0)
        self.cell_is_leaf = np.asarray(is_leaf, dtype=bool)
        self._node_particles = node_particles

        # Dense, indexable leaf -> particle membership for near-field direct sums.
        # Build (leaf_offset, leaf_count, leaf_particles) in one pass over leaves.
        K = len(centers)
        leaf_offset = np.zeros(K, dtype=np.int64)
        leaf_count = np.zeros(K, dtype=np.int64)
        segments = []
        for node in range(K):
            if self.cell_is_leaf[node]:
                pids = self._node_particles.get(node, np.empty(0, dtype=np.int64))
                leaf_count[node] = pids.size
                segments.append(pids)
        cat = np.concatenate(segments) if segments else np.empty(0, dtype=np.int64)
        # Cumsum of counts gives each leaf's start offset into `cat`.
        pos_in_cat = 0
        for node in range(K):
            if self.cell_is_leaf[node]:
                leaf_offset[node] = pos_in_cat
                pos_in_cat += int(leaf_count[node])
        self._leaf_offset = leaf_offset
        self._leaf_count = leaf_count
        self._leaf_particles = cat

        leaf_of = np.empty(self.n, dtype=np.int64)
        for node, pids in node_particles.items():
            if self.cell_is_leaf[node]:
                leaf_of[pids] = node
        self.leaf_of = leaf_of

    # -- differentientiable moments (vectorised bottom-up) -----------------
    def moments(self, masses: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute every cell's monopole (mass), COM, and quadrupole bottom-up.

        The accumulators follow the dtype of ``masses`` rather than being pinned to float32. The
        shipped callers pass float32, so their behaviour is unchanged; but pinning caused a hard
        failure (``index_put_`` dtype mismatch) under ``torch.set_default_dtype(torch.float64)``,
        which is exactly what a precision study does. A float64 cell centre also matters for the
        order-refinement surplus, where the truncation error falls below float32 rounding.
        """
        masses = masses.reshape(-1)
        md = masses.dtype
        device = masses.device
        K = len(self.cell_center)

        leaf_of = torch.from_numpy(self.leaf_of).to(device).long()
        pts = torch.from_numpy(self.pos).to(device).to(md)

        mass_sum = torch.zeros(K, device=device, dtype=md)
        mom1 = torch.zeros(K, 3, device=device, dtype=md)
        mom2 = torch.zeros(K, 3, 3, device=device, dtype=md)

        # Scatter leaf contributions.
        mass_sum = mass_sum.index_put_((leaf_of,), masses, accumulate=True)
        mom1 = mom1.index_put_((leaf_of,), masses.unsqueeze(1) * pts, accumulate=True)
        outer = pts.unsqueeze(2) * pts.unsqueeze(1)  # (N,3,3)
        mom2 = mom2.index_put_((leaf_of,), masses.unsqueeze(1).unsqueeze(1) * outer, accumulate=True)

        # Aggregate bottom-up by depth level (vectorised; max_depth iterations).
        depth = torch.from_numpy(self.cell_depth).to(device)
        parents = torch.from_numpy(self.cell_parent).to(device).long()
        max_depth = int(depth.max().item())
        for d in range(max_depth, 0, -1):
            nodes_d = torch.nonzero(depth == d, as_tuple=False).squeeze(1)
            if nodes_d.numel() == 0:
                continue
            p = parents[nodes_d]
            p = p[p >= 0]
            if p.numel() == 0:
                continue
            mass_sum = mass_sum.index_add_(0, p, mass_sum[nodes_d])
            mom1 = mom1.index_add_(0, p, mom1[nodes_d])
            mom2 = mom2.index_add_(0, p, mom2[nodes_d])

        total_mass = mass_sum
        com = mom1 / (total_mass.unsqueeze(1) + 1e-12)
        outer_m1 = mom1.unsqueeze(2) * mom1.unsqueeze(1)
        quadrupole = mom2 - outer_m1 / (total_mass.unsqueeze(1).unsqueeze(2) + 1e-12)

        return {"com": com, "total_mass": total_mass, "quadrupole": quadrupole}

    def leaf_particles_of(self, cell_index: int) -> np.ndarray:
        """Particle indices in a leaf cell (empty for non-leaf cells)."""
        start = int(self._leaf_offset[cell_index])
        count = int(self._leaf_count[cell_index])
        return self._leaf_particles[start : start + count]

    def expand_leaf_particles(
        self, cells: np.ndarray, self_particles: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fully-vectorised expansion of leaf cell membership (no Python loop).

        Given ``cells`` (a 1D array of leaf-cell indices), returns:
            owner   (P,)  position (0..len(cells)-1) of the requested cell
            pid     (P,)  global particle index of each member
            mask    (P,)  bool, True where pid != self_particles[owner] (excludes the
                          node's own particle, i.e. self-interaction)
        """
        cells = np.asarray(cells, dtype=np.int64)
        counts = self._leaf_count[cells].astype(np.int64)
        starts = self._leaf_offset[cells]
        total = int(counts.sum())
        # Position (k) within the requested cells, repeated per particle.
        owner = np.repeat(np.arange(len(cells)), counts)
        # Per-particle flat index into _leaf_particles = start[k] + within.
        cum = np.cumsum(counts)
        within = np.arange(total) - np.repeat(cum - counts, counts)
        positions = np.repeat(starts, counts) + within
        pid = self._leaf_particles[positions]
        if self_particles is not None:
            mask = pid != self_particles[owner]
        else:
            mask = np.ones(total, dtype=bool)
        return owner, pid, mask

    # -- sparse far-field interaction list --------------------------------
    def build_interaction_list(
        self,
        theta: float = 0.5,
        device: torch.device | None = None,
        max_interactions: int | None = None,
        node_start: int | None = None,
        node_end: int | None = None,
        query_nodes: torch.Tensor | None = None,
        opening_mode: str = "center",
        com: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorised Barnes-Hut descent -> sparse far-field and near-leaf lists.

        For every node in ``[node_start, node_end)`` (default: all nodes) we descend
        the tree:
          * a cell that satisfies the opening criterion ``size / dist < theta`` is
            a **far-field** source (approximated by its multipole) -- returned in the
            first pair ``(far_src, far_cell)``; and
          * a *leaf* cell that does NOT satisfy the opening criterion is a
            **near-field** source: its member particles are summed directly -- these
            leaves are returned in the second pair ``(near_leaf_src, near_leaf_cell)``.

        This closes the "middle-distance" gap a naive split leaves halfway: every
        particle-to-neighbourhood interaction is either a far-field multipole or a
        near-field direct leaf sum, so nothing is dropped.

        ``query_nodes`` (optional) restricts the descent to an explicit set of query
        node indices instead of a contiguous range. This is what makes a brute-force
        exact-force comparison possible at multi-million ``N``: the exact sum is
        ``O(Q * N)`` for ``Q`` queries, so one can test the operator at full scale by
        querying a sample rather than every node.

        ``opening_mode`` selects the opening criterion:
          * ``"com"`` -- ``size / |x - com| < theta``, i.e. the distance to the cell's
            **centre of mass**, which is also the point the multipole expansion is taken
            about. This is the criterion as published (Barnes & Hut 1986, as restated by
            Becciani et al. 2000; GADGET-1; SWIFT's ``multipole_accept.h``), and it is the
            one production cosmology codes use. Requires ``com``.
          * ``"center"`` -- ``size / |x - cell_center| < theta``, where ``cell_center`` is
            the **geometric** centre of the octree cell. **This is NOT the classical
            criterion**: it measures the acceptance distance to a different point from the
            one the expansion is taken about, so admission does not bound the distance to
            the multipole's expansion centre. It is a more permissive test than ``"com"``,
            and it is what this repository's earlier results were measured with.
          * ``"surface"`` -- ``size / max(|x - cell_center| - (sqrt(3)/2) size, eps)``,
            i.e. the distance to the cell's circumscribed-sphere boundary. A particle
            inside a cell's circumsphere drives the denominator to ``eps``, so the cell
            can never be admitted and must be expanded (or, if a leaf, summed exactly).

        The distinction between ``"com"`` and ``"center"`` is decision-relevant: for
        ``"center"`` the minimum separation between a query particle and an admitted cell's
        expansion centre is only ``size * (1/theta - sqrt(3)/2)``, which vanishes at
        ``theta = 2/sqrt(3) ~ 1.155``; for ``"com"`` admission implies the particle is
        strictly outside the source's own bounding sphere for every ``theta < 1``.

        Returns ``(far_src, far_cell, near_leaf_src, near_leaf_cell)``.
        Memory is O(N log N) rather than O(N * K).
        """
        if opening_mode not in ("center", "surface", "com"):
            raise ValueError(
                f"opening_mode must be 'center', 'surface' or 'com', got {opening_mode!r}"
            )
        if opening_mode == "com" and com is None:
            raise ValueError("opening_mode='com' requires the cell centres of mass")
        device = device or torch.device("cpu")
        cell_center = torch.from_numpy(self.cell_center).to(device).float()
        com_t = com.to(device).float() if com is not None else None
        cell_size = torch.from_numpy(self.cell_size).to(device).float()
        children = torch.from_numpy(self.cell_children).to(device).long()
        is_leaf = torch.from_numpy(self.cell_is_leaf).to(device)
        pos = torch.from_numpy(self.pos).to(device).float()

        start = node_start if node_start is not None else 0
        end = node_end if node_end is not None else self.n
        if query_nodes is not None:
            src = query_nodes.to(device).long().reshape(-1)
            cell = torch.zeros(src.numel(), dtype=torch.long, device=device)  # root = 0
        else:
            if start >= end:
                return (
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device),
                )
            src = torch.arange(start, end, device=device)
            cell = torch.zeros(end - start, dtype=torch.long, device=device)  # root = 0

        out_src: list[torch.Tensor] = []
        out_cell: list[torch.Tensor] = []
        near_src: list[torch.Tensor] = []
        near_cell: list[torch.Tensor] = []
        total_kept = 0

        while src.numel() > 0:
            if opening_mode == "com":
                # Classical Barnes-Hut: distance to the cell's CENTRE OF MASS, i.e. to the
                # point the multipole is expanded about. Admission then implies the query
                # particle lies strictly outside the source's bounding sphere for theta < 1.
                diff = pos[src] - com_t[cell]  # (M,3)
            else:
                # Distance to the GEOMETRIC cell centre. This is the criterion this
                # repository's earlier results used; it is not the published one, and it
                # bounds a different point from the expansion centre.
                diff = pos[src] - cell_center[cell]  # (M,3)
            dist = diff.norm(dim=-1) + 1e-8  # (M,)
            if opening_mode == "surface":
                # Distance to the cell, not to its centre: subtract the
                # circumscribed-sphere radius (sqrt(3)/2 * side) and clamp, so a cell
                # that could contain the query particle is never admitted.
                d_eff = (dist - 0.8660254037844386 * cell_size[cell]).clamp_min(1e-8)
            else:
                d_eff = dist
            opening = (cell_size[cell] / d_eff) < theta  # (M,)

            # Far field: keep cells that open.
            if opening.any():
                out_src.append(src[opening])
                out_cell.append(cell[opening])
                total_kept += int(opening.sum())

            # Near field: a leaf that does NOT open is a direct-sum source.
            near_leaf = (~opening) & is_leaf[cell]
            if near_leaf.any():
                near_src.append(src[near_leaf])
                near_cell.append(cell[near_leaf])
                total_kept += int(near_leaf.sum())

            # Stop if we've kept too many interactions (memory cap).
            if max_interactions is not None and total_kept >= max_interactions:
                break

            # Otherwise expand non-leaf, non-opening cells.
            expand = (~opening) & (~is_leaf[cell])
            e_src = src[expand]
            e_cell = cell[expand]
            if e_src.numel() == 0:
                break

            ch = children[e_cell]  # (E,8)
            src_x = e_src.unsqueeze(1).expand(-1, 8).reshape(-1)  # (E*8,)
            cell_x = ch.reshape(-1)
            keep = cell_x >= 0
            src = src_x[keep]
            cell = cell_x[keep]

        def _cat(parts: list[torch.Tensor]) -> torch.Tensor:
            if parts:
                return torch.cat(parts)
            return torch.empty(0, dtype=torch.long, device=device)

        return _cat(out_src), _cat(out_cell), _cat(near_src), _cat(near_cell)


def iter_interaction_batches(tree: "BarnesHutTree", theta: float = 0.5, device=None,
                            chunk_size: int = 5_000_000, node_start: int = 0, node_end: int | None = None):
    """Yield `(far_src, far_cell, near_src, near_cell)` in batches of <= chunk_size.

    Batches the Barnes-Hut interaction list into memory-bounded chunks so the FULL
    (uncapped) interaction list can be streamed through the far/near field without
    ever storing all entries at once (a 'streaming iterator'). Each batch's entries
    are concatenated and returned; the caller accumulates the resulting
    accelerations and discards the batch.

    NOTE: the underyling tree descent still touches every node, but the returned
    tensors are capped at ~chunk_size entries, bounding peak memory.
    """
    device = device or torch.device("cpu")
    node_end = tree.n if node_end is None else node_end

    # Collect the whole (tagged) frontier with one full descent, then yield it in
    # memory-bounded chunks. For a truly streaming builder we would interleave the
    # descent with yield; here we cap the *returned* batch so the caller never holds
    # more than chunk_size per call (peak memory bounded) while the descent itself is
    # one vectorised pass.
    fs, fc, ns, nc = tree.build_interaction_list(
        theta=theta, device=device, node_start=node_start, node_end=node_end
    )
    # Interleave far/near into tagged batches by slicing both lists in step chunks.
    total = fs.numel() + ns.numel()
    for b in range(0, max(fs.numel(), ns.numel()), chunk_size):
        f_s = fs[b : b + chunk_size]
        f_c = fc[b : b + chunk_size]
        n_s = ns[b : b + chunk_size] if b < ns.numel() else torch.empty(0, dtype=torch.long, device=device)
        n_c = nc[b : b + chunk_size] if b < ns.numel() else torch.empty(0, dtype=torch.long, device=device)
        if f_s.numel() + n_s.numel() > 0:
            yield f_s, f_c, n_s, n_c


def multipole_moments_from_tree(
    tree: BarnesHutTree, masses: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Convenience wrapper around :meth:`BarnesHutTree.moments`."""
    return tree.moments(masses)


def exact_force(
    positions: torch.Tensor,
    masses: torch.Tensor,
    softening: float = 1e-4,
) -> torch.Tensor:
    """Exact pairwise gravitational acceleration (O(N^2)), for the error bound.

    Standard Newtonian convention:
    ``a_i = G * sum_j m_j (x_j - x_i) / (|x_ij|^2 + s^2)^{3/2}``.
    """
    G = 1.0
    pos = positions
    # diff[i, j] = x_j - x_i  (so particle i is pulled toward particle j).
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N,N,3)
    dist_sq = (diff * diff).sum(-1) + softening**2
    inv_r3 = dist_sq.rsqrt() ** 3
    acc = torch.einsum("j,ij,ijI->iI", masses.float(), inv_r3, diff)
    return G * acc
