"""HMMP: Hierarchical Multipole Message Passing model.

The model partitions each node's interactions into:
  * **near field**  -- direct pairwise gravitational messages on graph edges, and
  * **far field**   -- multipole-approximated contributions from octree cell centres.

The far field uses a **sparse** Barnes-Hut interaction list (O(N log N) memory)
instead of a dense ``(N, K)`` matrix, so HMMP scales to millions of nodes. Both
fields are differentiable: the tree structure is fixed by the positions, while the
multipole moments (monopole + COM, optionally quadrupole) depend on node masses
through which gradients flow.

The Barnes-Hut tree is cached by ``(position signature)``: it is rebuilt only when
the input positions actually change, which is never across training epochs for a
fixed snapshot.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .barnes_hut import BarnesHutTree
from .gravity import GravitationalMessage
from .grad_utils import compute_acceleration


class HMMPLayer(nn.Module):
    """One HMMP message-passing layer: near field (edges) + far field (multipole)."""

    def __init__(
        self,
        node_dim: int = 2,
        hidden_dim: int = 64,
        use_learned: bool = True,
        softening: float = 1e-4,
        use_quadrupole: bool = False,
        far_batch: int = 1 << 20,
        edge_chunk: int | None = None,
    ):
        super().__init__()
        self.use_quadrupole = use_quadrupole
        self.far_batch = far_batch
        self.edge_chunk = edge_chunk
        self.gravity = GravitationalMessage(
            node_dim=node_dim, hidden_dim=hidden_dim, use_learned=use_learned, softening=softening
        )
        # New node features: [old_feats | physical_acc(3) | old_feats].
        in_dim = node_dim + 3 + node_dim
        self.update = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def far_field_acceleration(
        self,
        tree: BarnesHutTree,
        moments: dict[str, torch.Tensor],
        pos: torch.Tensor,
        mass: torch.Tensor,
        theta: float = 0.5,
        interaction_list: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        node_start: int | None = None,
        node_end: int | None = None,
    ) -> torch.Tensor:
        """Sparse multipole far-field acceleration per node (O(N log N) memory).

        Only cells that satisfy the Barnes-Hut opening criterion contribute, gathered
        from the precomputed interaction list, in bounded batches.
        """
        device = pos.device
        com = moments["com"].to(device)
        total_mass = moments["total_mass"].to(device)

        if interaction_list is None:
            far_src, far_cell, _ns, _nc = tree.build_interaction_list(theta=theta, device=device)
        else:
            far_src, far_cell = interaction_list[0], interaction_list[1]
            far_src = far_src.to(device)
            far_cell = far_cell.to(device)

        n = pos.shape[0]
        far_acc = torch.zeros(n, 3, device=device)

        if self.use_quadrupole:
            quad = moments["quadrupole"].to(device)

        for b in range(0, far_src.numel(), self.far_batch):
            s = far_src[b : b + self.far_batch]
            c = far_cell[b : b + self.far_batch]
            if s.numel() == 0:
                break
            # diff = (cell COM) - (particle) = x_c - x_i  (standard attraction sign:
            # particle i is pulled toward the cell's centre of mass).
            diff = com[c] - pos[s]  # (B,3)
            dist_sq = (diff * diff).sum(-1) + 1e-8  # (B,)
            dist = dist_sq.sqrt()
            inv_r3 = dist_sq.rsqrt() ** 3
            mono = total_mass[c].unsqueeze(-1) * diff * inv_r3.unsqueeze(-1)  # (B,3)

            if self.use_quadrupole:
                # Far field from the traceless quadrupole tensor
                #   I_ij = sum_a m_a (3 y_a,i y_a,j - |y_a|^2 delta_ij) = 3D - tr(D) Id,
                # where D = sum_a m_a y_a y_a^T is the second moment about the cell COM
                # (that is what BarnesHutTree.moments returns). With rhat = (com - x)/r
                # and n = -rhat, the potential Phi = -G[M/r + (1/2) I_ij n_i n_j / r^3]
                # gives
                #   a_quad = (1/2)[ -6 (D rhat) - 3 tr(D) rhat
                #                   + 15 (rhat^T D rhat) rhat ] / r^4 .
                # The 1/r^4 decay (not 1/r^3) and the (D rhat) term are both required;
                # dropping either makes the term larger than the monopole residual it is
                # meant to cancel.
                rhat = diff / dist.unsqueeze(-1)
                inv_r4 = inv_r3 / dist
                quad_c = quad[c]
                tr_d = quad_c.diagonal(dim1=-2, dim2=-1).sum(-1)          # (B,)
                d_rhat = torch.einsum("bIJ,bJ->bI", quad_c, rhat)         # (B,3)
                r_d_r = torch.einsum("bI,bI->b", d_rhat, rhat)            # (B,)
                a_quad = (
                    -6.0 * d_rhat
                    - 3.0 * tr_d.unsqueeze(-1) * rhat
                    + 15.0 * r_d_r.unsqueeze(-1) * rhat
                )
                mono = mono + 0.5 * a_quad * inv_r4.unsqueeze(-1)

            far_acc = far_acc.index_add_(0, s, mono)

        return far_acc

    def far_field_acceleration_and_tidal(
        self,
        tree: BarnesHutTree,
        moments: dict[str, torch.Tensor],
        pos: torch.Tensor,
        mass: torch.Tensor,
        theta: float = 0.5,
        interaction_list: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Far-field multipole acceleration **and** its derivative, in one pass.

        Returns ``(far_acc, far_tidal)`` with ``far_acc`` identical to what
        :meth:`far_field_acceleration` returns for the same arguments, and

            far_tidal[i, a, b] = d far_acc[i, a] / d pos[i, b]   (the tidal tensor).

        Convention. With ``d = com_cell - x_i``, ``r^2 = |d|^2 + 1e-8`` and ``rhat = d/r``:

          * monopole (p = 0), from ``a = M d / r^3``:
                ``T = M [ 3 rhat rhat^T - I ] / r^3``
          * quadrupole (p = 1), one more derivative of the same ``a_quad`` used above.
            Differentiating the bracket exactly (the 1/r^4 radial factor contributes too)
            gives, with ``D`` the second moment about the cell COM, ``trD`` its trace and
            ``Q = rhat^T D rhat``:

                ``T_quad = (1/2) [ 6 D + (3 trD - 15 Q) I
                                   - 30 ( (D rhat) rhat^T + rhat (D rhat)^T )
                                   - (15 trD - 105 Q) rhat rhat^T ] / r^5``

            The ``-30`` cross term and the ``+105 Q`` are what make this symmetric and
            reproduce the analytic ``-6 m h^2 / R^4`` quadrupole increment's derivative
            ``+24 m h^2 / R^5`` for the two-mass cell of ``tests/test_quadrupole_physics``.
            ``tests/test_tidal_tensor.py`` checks the whole expression against
            ``torch.autograd`` rather than against a hand-copied formula.

        Note the near/far split is unchanged: only the multipole derivative count differs,
        so the near-leaf contribution is identical for every ``p``.
        """
        device = pos.device
        com = moments["com"].to(device)
        total_mass = moments["total_mass"].to(device)

        if interaction_list is None:
            far_src, far_cell, _ns, _nc = tree.build_interaction_list(theta=theta, device=device)
        else:
            far_src, far_cell = interaction_list[0], interaction_list[1]
            far_src = far_src.to(device)
            far_cell = far_cell.to(device)

        n = pos.shape[0]
        far_acc = torch.zeros(n, 3, device=device, dtype=pos.dtype)
        far_tide = torch.zeros(n, 3, 3, device=device, dtype=pos.dtype)
        eye = torch.eye(3, device=device, dtype=pos.dtype)

        if self.use_quadrupole:
            quad = moments["quadrupole"].to(device)

        for b in range(0, far_src.numel(), self.far_batch):
            s = far_src[b : b + self.far_batch]
            c = far_cell[b : b + self.far_batch]
            if s.numel() == 0:
                break
            # diff = (cell COM) - (particle) = x_c - x_i  (standard attraction sign)
            diff = com[c] - pos[s]  # (B,3)
            dist_sq = (diff * diff).sum(-1) + 1e-8  # (B,)
            dist = dist_sq.sqrt()
            inv_r3 = dist_sq.rsqrt() ** 3
            inv_r5 = inv_r3 / dist_sq
            mono = total_mass[c].unsqueeze(-1) * diff * inv_r3.unsqueeze(-1)  # (B,3)

            outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)                  # (B,3,3)
            tide = total_mass[c].unsqueeze(-1).unsqueeze(-1) * (
                3.0 * outer * inv_r5.unsqueeze(-1).unsqueeze(-1)
                - eye * inv_r3.unsqueeze(-1).unsqueeze(-1)
            )

            if self.use_quadrupole:
                # Far field from the second moment D about the cell COM (see
                # far_field_acceleration for why the 1/r^4 power and the (D rhat) term
                # are both required).
                rhat = diff / dist.unsqueeze(-1)
                inv_r4 = inv_r3 / dist
                quad_c = quad[c]
                tr_d = quad_c.diagonal(dim1=-2, dim2=-1).sum(-1)          # (B,)
                d_rhat = torch.einsum("bIJ,bJ->bI", quad_c, rhat)         # (B,3)
                r_d_r = torch.einsum("bI,bI->b", d_rhat, rhat)            # (B,)
                a_quad = (
                    -6.0 * d_rhat
                    - 3.0 * tr_d.unsqueeze(-1) * rhat
                    + 15.0 * r_d_r.unsqueeze(-1) * rhat
                )
                mono = mono + 0.5 * a_quad * inv_r4.unsqueeze(-1)

                # ... and its exact spatial derivative (the same bracket, differentiated).
                rhat_rhat = rhat.unsqueeze(-1) * rhat.unsqueeze(-2)       # (B,3,3)
                cross = (
                    d_rhat.unsqueeze(-1) * rhat.unsqueeze(-2)
                    + rhat.unsqueeze(-1) * d_rhat.unsqueeze(-2)
                )                                                         # (B,3,3)
                t_quad = 0.5 * inv_r5.unsqueeze(-1).unsqueeze(-1) * (
                    6.0 * quad_c
                    + (3.0 * tr_d - 15.0 * r_d_r).unsqueeze(-1).unsqueeze(-1) * eye
                    - 30.0 * cross
                    - (15.0 * tr_d - 105.0 * r_d_r).unsqueeze(-1).unsqueeze(-1) * rhat_rhat
                )
                tide = tide + t_quad

            far_acc = far_acc.index_add_(0, s, mono)
            far_tide = far_tide.index_add_(0, s, tide)

        return far_acc, far_tide

    def near_leaf_acceleration(
        self,
        tree: BarnesHutTree,
        pos: torch.Tensor,
        mass: torch.Tensor,
        interaction_list: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        theta: float = 0.5,
    ) -> torch.Tensor:
        """Direct (exact) summation over leaf cells that did NOT satisfy the opening
        criterion. This closes the middle-distance gap that the far-field multipole
        leaves open, so near+far together approximate the exact pairwise force.

        For each node we sum ``m_j * (x_j - x_i) / |x_ij|^3`` over the member
        particles of each near leaf cell. Done in bounded batches.
        """
        device = pos.device
        if interaction_list is None:
            _fs, _fc, near_src, near_cell = tree.build_interaction_list(theta=theta, device=device)
        else:
            near_src, near_cell = interaction_list[2], interaction_list[3]
            near_src = near_src.to(device)
            near_cell = near_cell.to(device)

        n = pos.shape[0]
        near_acc = torch.zeros(n, 3, device=device)

        # For each near-leaf interaction we need the leaf's member particles. Use the
        # vectorised expansion (one gather per batch, no Python loop) so it scales to
        # millions of nodes.
        for b in range(0, near_src.numel(), self.far_batch):
            s = near_src[b : b + self.far_batch]
            c = near_cell[b : b + self.far_batch]
            if s.numel() == 0:
                break
            s_np = s.detach().cpu().numpy()
            c_np = c.detach().cpu().numpy()
            owner, pid, mask = tree.expand_leaf_particles(c_np, self_particles=s_np)
            if not mask.any():
                continue
            owner = owner[mask]
            pid = pid[mask]
            n_idx = s_np[owner]
            n_idx_t = torch.from_numpy(n_idx).to(device)
            p_idx_t = torch.from_numpy(pid).to(device)
            d = pos[p_idx_t] - pos[n_idx_t]  # (P,3) = x_j - x_i
            dsq = (d * d).sum(-1) + 1e-8
            direct = mass[p_idx_t].unsqueeze(-1) * d * (dsq.rsqrt() ** 3).unsqueeze(-1)
            near_acc = near_acc.index_add_(0, n_idx_t, direct)

        return near_acc

    def near_leaf_acceleration_and_tidal(
        self,
        tree: BarnesHutTree,
        pos: torch.Tensor,
        mass: torch.Tensor,
        interaction_list: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        theta: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Near-leaf direct sums of the acceleration **and** its derivative.

        Identical to :meth:`near_leaf_acceleration` for the acceleration; the second
        return value is the exact tidal tensor of the same direct pairwise sum,

            ``T_ij = sum_p m_p [ 3 d_i d_j / r^5 - delta_ij / r^3 ]``,  ``d = x_p - x_i``.

        Because this sum is exact it carries no multipole order, so it contributes
        identically to ``p = 0`` and ``p = 1`` -- it cancels in any order-to-order ratio.
        """
        device = pos.device
        if interaction_list is None:
            _fs, _fc, near_src, near_cell = tree.build_interaction_list(theta=theta, device=device)
        else:
            near_src, near_cell = interaction_list[2], interaction_list[3]
            near_src = near_src.to(device)
            near_cell = near_cell.to(device)

        n = pos.shape[0]
        near_acc = torch.zeros(n, 3, device=device, dtype=pos.dtype)
        near_tide = torch.zeros(n, 3, 3, device=device, dtype=pos.dtype)
        eye = torch.eye(3, device=device, dtype=pos.dtype)

        for b in range(0, near_src.numel(), self.far_batch):
            s = near_src[b : b + self.far_batch]
            c = near_cell[b : b + self.far_batch]
            if s.numel() == 0:
                break
            s_np = s.detach().cpu().numpy()
            c_np = c.detach().cpu().numpy()
            owner, pid, mask = tree.expand_leaf_particles(c_np, self_particles=s_np)
            if not mask.any():
                continue
            owner = owner[mask]
            pid = pid[mask]
            n_idx = s_np[owner]
            n_idx_t = torch.from_numpy(n_idx).to(device)
            p_idx_t = torch.from_numpy(pid).to(device)
            d = pos[p_idx_t] - pos[n_idx_t]  # (P,3) = x_j - x_i
            dsq = (d * d).sum(-1) + 1e-8
            inv_r3 = dsq.rsqrt() ** 3
            mw = mass[p_idx_t]
            direct = mw.unsqueeze(-1) * d * inv_r3.unsqueeze(-1)
            outer = d.unsqueeze(-1) * d.unsqueeze(-2)
            contrib = mw.unsqueeze(-1).unsqueeze(-1) * (
                3.0 * outer * (inv_r3 / dsq).unsqueeze(-1).unsqueeze(-1)
                - eye * inv_r3.unsqueeze(-1).unsqueeze(-1)
            )
            near_acc = near_acc.index_add_(0, n_idx_t, direct)
            near_tide = near_tide.index_add_(0, n_idx_t, contrib)

        return near_acc, near_tide

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mass: torch.Tensor,
        edge_index: torch.Tensor,
        tree: BarnesHutTree,
        moments: dict[str, torch.Tensor],
        theta: float = 0.5,
        interaction_list: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        node_start: int | None = None,
        node_end: int | None = None,
        edge_slice: torch.Tensor | None = None,
        process_group=None,
    ) -> torch.Tensor:
        """Layer forward over this rank's node shard.

        ``x`` is this rank's shard feature tensor ``[node_start, node_end)``. We:
          1. all_gather the shard features into the FULL node feature tensor (so
             cross-shard GNN neighbours resolve);
          2. compute the shard's GNN near-field (edges whose dst is in this shard);
          3. compute the shard's physical far/near field from the *sharded* interaction
             list;
          4. all_reduce the physical field so every rank sees the full-node physical
             field;
          5. update this rank's shard features.
        """
        import torch.distributed as dist

        start = 0 if node_start is None else node_start
        end = x.shape[0] if node_start is None else (node_end or node_start + x.shape[0])
        full_n = pos.shape[0]
        world = dist.get_world_size(process_group) if process_group is not None else 1

        if world > 1:
            gathered = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(gathered, x, group=process_group)
            h_full = torch.cat(gathered, dim=0)
        else:
            h_full = x  # single-GPU: shard == full

        # GNN near-field: only edges whose dst is this shard -> shard node values.
        if edge_slice is not None:
            edge_used = edge_slice
        else:
            edge_used = edge_index
        graph_full = compute_acceleration(
            self.gravity(h_full, pos, mass, edge_used, edge_chunk=self.edge_chunk), edge_used, full_n
        )
        # Physical far/near field: sharded interaction list, full-N output (only the
        # shard's interactions contribute to the shard's nodes).
        far_full = self.far_field_acceleration(tree, moments, pos, mass, theta=theta, interaction_list=interaction_list)
        leaf_full = self.near_leaf_acceleration(tree, pos, mass, theta=theta, interaction_list=interaction_list)
        physical_full = graph_full + leaf_full + far_full  # (full_n, 3)

        if world > 1:
            dist.all_reduce(physical_full, op=dist.ReduceOp.SUM, group=process_group)

        phys = physical_full[start:end]
        feat = torch.cat([x, phys, x], dim=-1)
        updated = self.update(feat)  # (shard, hidden)
        return updated


class HMMP(nn.Module):
    """Full HMMP network: build tree -> N layers of near/far message passing -> density head."""

    def __init__(
        self,
        node_dim: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        max_leaf: int = 16,
        max_depth: int = 24,
        theta: float = 0.5,
        use_learned: bool = True,
        use_quadrupole: bool = False,
        softening: float = 1e-4,
        far_batch: int = 1 << 20,
        edge_chunk: int | None = None,
        use_checkpoint: bool = False,
        max_interactions: int | None = None,
        use_transformer: bool = False,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        legacy_geometry: bool = False,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.n_layers = n_layers
        self.theta = theta
        self.max_leaf = max_leaf
        self.max_depth = max_depth
        # Passed straight to BarnesHutTree. False (the default) is the corrected geometry; True
        # reproduces defect D8 exactly, so the cost of the fix can be measured against the same
        # machine rather than against a number from an older tree. See scripts/compare_d8_cost_ab.py.
        self.legacy_geometry = bool(legacy_geometry)
        self.use_learned = use_learned
        self.use_quadrupole = use_quadrupole
        self.edge_chunk = edge_chunk
        self.use_checkpoint = use_checkpoint
        self.max_interactions = max_interactions
        self.use_transformer = use_transformer
        self._process_group = None

        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList(
            [
                HMMPLayer(
                    node_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    use_learned=use_learned,
                    softening=softening,
                    use_quadrupole=use_quadrupole,
                    far_batch=far_batch,
                    edge_chunk=edge_chunk,
                )
                for _ in range(n_layers)
            ]
        )
        # Optional sparse graph-attention stack (HMMP + Sparse Transformer hybrid):
        # the BH multipole / GNN encode the physical field, then shallow graph attention
        # refines cross-node semantics along the same sparse edges (O(N log N)).
        from .sparse_attn import SparseGraphAttention

        self.attn_layers = nn.ModuleList(
            [SparseGraphAttention(hidden_dim, heads=transformer_heads, edge_chunk=edge_chunk)
             for _ in range(transformer_layers)]
        ) if use_transformer else nn.ModuleList()
        # Project the physically-injected token ([hidden | far_acc(3)] -> hidden) so the
        # attention Q/K/V see the Newtonian multipole field as an explicit bias.
        self.phys_proj = nn.Linear(hidden_dim + 3, hidden_dim) if use_transformer else None
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Cache the tree + interaction list so we never rebuild them across epochs
        # of the same (identical) position tensor. Keyed by data_ptr + shape + size:
        # O(1), no device sync, correct because the trainer reuses the same tensor.
        self._cache_ptr: int | None = None
        self._cache_shape: tuple | None = None
        self._cache_size: float | None = None
        self._cache_theta: float | None = None
        self._cache_pos_ref: torch.Tensor | None = None
        self._tree: BarnesHutTree | None = None
        self._interaction: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def build_tree(self, pos: torch.Tensor, center: torch.Tensor, size: float) -> BarnesHutTree:
        tree = BarnesHutTree(
            pos,
            center=center,
            size=size,
            max_leaf=self.max_leaf,
            max_depth=self.max_depth,
            legacy_geometry=self.legacy_geometry,
        )
        return tree

    def _get_tree_and_list(
        self, pos: torch.Tensor, center: torch.Tensor, size: float
    ) -> tuple[BarnesHutTree, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Keep a reference to the cache key so the tensor is not garbage-collected
        # and its data_ptr is not reused by a different tensor.
        same = (
            self._tree is not None
            and self._cache_ptr == pos.data_ptr()
            and self._cache_shape == tuple(pos.shape)
            and self._cache_size == size
            and self._cache_theta == self.theta
        )
        if same:
            return self._tree, self._interaction  # type: ignore[return-value]
        tree = self.build_tree(pos, center=center, size=size)
        interaction = tree.build_interaction_list(
            theta=self.theta, device=pos.device, max_interactions=self.max_interactions
        )
        self._tree = tree
        self._interaction = interaction
        self._cache_ptr = pos.data_ptr()
        self._cache_shape = tuple(pos.shape)
        self._cache_size = size
        self._cache_theta = self.theta
        self._cache_pos_ref = pos
        return tree, interaction

    def forward(
        self,
        pos: torch.Tensor,
        mass: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        center: torch.Tensor,
        size: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(pred_log_density, far_field_acc, n_interactions)`` for ALL nodes."""
        return self.forward_node_range(
            pos, mass, x, edge_index, center, size, node_start=0, node_end=pos.shape[0]
        )

    def forward_node_range(
        self,
        pos: torch.Tensor,
        mass: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        center: torch.Tensor,
        size: float,
        node_start: int,
        node_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pred, far_acc, n_int) for the node slice ``[node_start, node_end)``.

        The tree and global moments are cached / computed from ALL masses (replicated
        across ranks; it is small), while the Barnes-Hut interaction list is built only
        for the requested node slice. This is the memory-sharding primitive used by the
        multi-GPU path (per-rank interaction list is O(slice log N) instead of O(N log N)).
        """
        tree = self._get_tree(pos, center, size)
        moments = tree.moments(mass)  # global moments (all masses), small to replicate

        # Build / reuse the interaction list restricted to [node_start, node_end).
        key = (node_start, node_end)
        if not hasattr(self, "_slice_cache") or self._slice_cache.get("key") != (pos.data_ptr(), node_start, node_end):
            interaction = tree.build_interaction_list(
                theta=self.theta, device=pos.device, max_interactions=self.max_interactions,
                node_start=node_start, node_end=node_end,
            )
            self._slice_cache = {"key": (pos.data_ptr(), node_start, node_end), "interaction": interaction}
        else:
            interaction = self._slice_cache["interaction"]

        # Shard-local node features: GNN + physics operate on this rank's node slice.
        x_shard = x[node_start:node_end]
        h = self.encoder(x_shard)

        # Restrict GNN messages to edges whose dst is in this shard (memory / world).
        if node_start > 0 or node_end < pos.shape[0]:
            dst_all = edge_index[1]
            in_shard = (dst_all >= node_start) & (dst_all < node_end)
            edge_slice = edge_index[:, in_shard]
        else:
            edge_slice = None

        last_far = None
        for layer in self.layers:
            if self.use_checkpoint:
                def _run_layer(layer=layer, h=h):
                    return layer(
                        h, pos, mass, edge_index, tree, moments,
                        theta=self.theta, interaction_list=interaction, node_start=node_start, node_end=node_end,
                        edge_slice=edge_slice, process_group=self._process_group,
                    )
                h = checkpoint(_run_layer, use_reentrant=False)
            else:
                h = layer(
                    h, pos, mass, edge_index, tree, moments,
                    theta=self.theta, interaction_list=interaction, node_start=node_start, node_end=node_end,
                    edge_slice=edge_slice, process_group=self._process_group,
                )
            last_far = layer.far_field_acceleration(
                tree, moments, pos, mass, theta=self.theta, interaction_list=interaction,
            )

        # Optional: HMMP + Sparse Transformer hybrid — refine semantics along the BH /
        # kNN sparse edges after the physical field has been encoded. The attention
        # needs FULL node features (its edges reference global node indices), so
        # all_gather the shard features first, then refine, then slice back to shard.
        if self.use_transformer:
            import torch.distributed as dist
            world = dist.get_world_size(self._process_group) if (self._process_group is not None and dist.is_initialized()) else 1
            if world > 1:
                gathered = [torch.empty_like(h) for _ in range(world)]
                dist.all_gather(gathered, h, group=self._process_group)
                h_full = torch.cat(gathered, dim=0)
            else:
                h_full = h
            # Physical context fusion: inject the far-field acceleration (already full-N,
            # with only this shard's nodes having a non-zero value) into the attention
            # token, so Q/K/V see the Newtonian multipole field as an explicit bias.
            if last_far is not None:
                phys_full = torch.nan_to_num(last_far, nan=0.0, posinf=0.0, neginf=0.0)
                h_full = self.phys_proj(torch.cat([h_full, phys_full], dim=-1))  # (full, hidden)
            for attn in self.attn_layers:
                if self.use_checkpoint:
                    def _run_attn(attn=attn, hh=h_full):
                        return attn(hh, pos, edge_index, edge_slice=edge_slice)
                    h_full = checkpoint(_run_attn, use_reentrant=False)
                else:
                    h_full = attn(h_full, pos, edge_index, edge_slice=edge_slice)
            h = h_full[node_start:node_end]

        # ``h`` is already shard-sized (each layer updates only this rank's slice).
        pred_log_density = self.head(h).squeeze(-1)
        n_int = interaction[0].numel()
        last_far_shard = last_far[node_start:node_end] if last_far is not None else None
        return pred_log_density, last_far_shard, torch.tensor(n_int, dtype=torch.float32, device=pos.device)

    def _get_tree(self, pos: torch.Tensor, center: torch.Tensor, size: float) -> BarnesHutTree:
        """Cached/global tree build (positions are replicated across ranks)."""
        if self._tree is not None and self._cache_ptr == pos.data_ptr() and self._cache_shape == tuple(pos.shape) and self._cache_size == size:
            return self._tree
        tree = self.build_tree(pos, center=center, size=size)
        self._tree = tree
        self._cache_ptr = pos.data_ptr()
        self._cache_shape = tuple(pos.shape)
        self._cache_size = size
        return tree

        pred_log_density = self.head(h).squeeze(-1)
        n_int = interaction[0].numel()
        return pred_log_density, last_far, torch.tensor(n_int, dtype=torch.float32, device=pos.device)
