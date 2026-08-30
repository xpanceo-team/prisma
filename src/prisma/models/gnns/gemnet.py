import os
from typing import Dict, Optional, Union, Callable

import torch
from torch import nn
from torch_scatter import scatter, scatter_add
from prisma.backbones.common.utils import conditional_grad
from prisma.backbones.gemnet import GemNetT
from prisma.backbones.gemnet.layers.embedding_block import (
    AtomEmbedding,
)
from prisma.backbones.gemnet.utils import inner_product_normalized
from prisma.backbones.gemnet.layers.base_layers import Dense

from diffusers.configuration_utils import register_to_config

from prisma.models.modeling_utils import ModelMixin
from prisma.configuration_utils import ConfigMixin
from prisma.models.embeddings import ControlNetAdapter


class AtomEmbeddingWrapper(AtomEmbedding):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(
            Z
        )  # Z-1 in original implementaion, but we use 0 as an absorbing state
        return h


class RBFBasedLatticeUpdateBlock(torch.nn.Module):
    # Lattice update block that mimics GemNet's edge processing, e.g., uses radial basis functions.
    def __init__(
        self,
        emb_size: int,
        activation: str,
        emb_size_rbf: int,
        emb_size_edge: int,
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_out = num_heads
        self.mlp = nn.Sequential(
            Dense(emb_size, emb_size, activation=activation), Dense(emb_size, emb_size)
        )
        self.dense_rbf_F = Dense(
            emb_size_rbf, emb_size_edge, activation=None, bias=False
        )
        self.out_forces = Dense(emb_size_edge, num_heads, bias=False, activation=None)

    def compute_score_per_edge(
        self,
        edge_emb: torch.Tensor,  # [Num_edges, emb_dim]
        rbf: torch.Tensor,  # [Num_edges, num_rbf_bases]
    ) -> torch.Tensor:
        x_F = self.mlp(edge_emb)
        rbf_emb_F = self.dense_rbf_F(rbf)  # (nEdges, emb_size_edge)
        x_F_rbf = x_F * rbf_emb_F
        # x_F = self.scale_rbf_F(x_F, x_F_rbf)
        x_F = self.out_forces(x_F_rbf)  # (nEdges, self.num_out)
        return x_F


def edge_score_to_lattice_score_frac_symmetric(
    score_d: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Converts a score per edge into a score for the atom coordinates and/or the lattice matrix via the chain rule.
    This method explicitly takes into account the fact that the cartesian coordinates depend on the lattice via the fractional coordinates.
    Moreover, we make sure to get a symmetric update: D_cart_norm @ Phi @ D_cart_norm^T, where Phi is a |E| x |E| diagonal matrix with the predicted edge scores

    Args:
        score_d (torch.Tensor, [num_edges,]): A score per edge in the graph.
        edge_index (torch.Tensor, [2, num_edges]): The edge indices in the graph.
        edge_vectors (torch.Tensor, [num_edges, 3]): The vectors connecting the source of each edge to the target.
        lattice_matrix (torch.Tensor, [num_nodes, 3, 3]): The lattice matrices for each crystal in num_nodes.
        batch (torch.Tensor, [num_nodes,]): The pointer indicating for each atom which molecule in the batch it belongs to.

    Returns:
        torch.Tensor: The predicted lattice score.
    """
    batch_edge = batch[edge_index[0]]
    unit_edge_vectors_cart = edge_vectors / edge_vectors.norm(dim=-1, keepdim=True)
    score_lattice = scatter_add(
        score_d[:, None, None]
        * (unit_edge_vectors_cart[:, :, None] @ unit_edge_vectors_cart[:, None, :]),
        batch_edge,
        dim=0,
        dim_size=batch.max() + 1,
    ).transpose(-1, -2)
    return score_lattice


class RBFBasedLatticeUpdateBlockFrac(RBFBasedLatticeUpdateBlock):
    # Lattice update block that mimics GemNet's edge processing, e.g., uses radial basis functions.
    def __init__(
        self,
        emb_size: int,
        activation: str,
        emb_size_rbf: int,
        emb_size_edge: int,
        num_heads: int = 1,
    ):
        super().__init__(
            emb_size=emb_size,
            activation=activation,
            emb_size_rbf=emb_size_rbf,
            emb_size_edge=emb_size_edge,
            num_heads=num_heads,
        )

    def forward(
        self,
        edge_emb: torch.Tensor,  # [Num_edges, emb_dim]
        edge_index: torch.Tensor,  # [2, Num_edges]
        distance_vec: torch.Tensor,  # [Num_edges, 3]
        lattice: torch.Tensor,  # [Num_crystals, 3, 3]
        batch: torch.Tensor,  # [Num_atoms, ]
        rbf: torch.Tensor,  # [Num_edges, num_rbf_bases]
        normalize_score: bool = True,
    ) -> torch.Tensor:
        edge_scores = self.compute_score_per_edge(edge_emb=edge_emb, rbf=rbf)
        if normalize_score:
            num_edges = scatter(
                torch.ones_like(distance_vec[:, 0]), batch[edge_index[0]]
            )
            edge_scores /= num_edges[batch[edge_index[0]], None]
        outs = []
        for i in range(self.num_out):
            lattice_update = edge_score_to_lattice_score_frac_symmetric(
                score_d=edge_scores[:, i],
                edge_index=edge_index,
                edge_vectors=distance_vec,
                batch=batch,
            )
            outs.append(lattice_update)
        outs = torch.stack(outs, dim=-1).sum(-1)
        # [Batch_size, 3, 3]
        return outs


class GemNetTWrapper(GemNetT, ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        original_lattice_update: bool = False,
        t_emb_dim: int = 512,
        condition_keys: list[str] | None = None,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        atom_emb_dim: int = 512,
        edge_emb_dim: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        regress_forces: bool = True,
        direct_forces: bool = True,
        rbf: Dict[str, any] | None = None,
        envelope: Dict[str, any] | None = None,
        cbf: Dict[str, any] | None = None,
        extensive: bool = False,
        use_pbc: bool = True,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        max_atomic_number: int = 100,
        scale_file: Optional[str] = None,
    ):
        if rbf is None:
            rbf = {"name": "gaussian"}

        if envelope is None:
            envelope = {"name": "polynomial", "exponent": 5}

        if cbf is None:
            cbf = {"name": "spherical_harmonics"}

        if condition_keys is None:
            condition_keys = []

        self.register_to_config(
            rbf=rbf, envelope=envelope, cbf=cbf, condition_keys=condition_keys
        )

        super().__init__(
            num_spherical=num_spherical,
            num_radial=num_radial,
            num_blocks=num_blocks,
            emb_size_atom=atom_emb_dim,
            emb_size_edge=edge_emb_dim,
            emb_size_trip=emb_size_trip,
            emb_size_rbf=emb_size_rbf,
            emb_size_cbf=emb_size_cbf,
            emb_size_bil_trip=emb_size_bil_trip,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_concat=num_concat,
            num_atom=num_atom,
            regress_forces=regress_forces,
            direct_forces=direct_forces,
            rbf=rbf,
            envelope=envelope,
            cbf=cbf,
            extensive=extensive,
            use_pbc=use_pbc,
            output_init=output_init,
            activation=activation,
            num_elements=max_atomic_number,
            scale_file=scale_file,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            otf_graph=True,  # set True and then after set new value to bypass assert
        )
        self.otf_graph = False  # we build graph ourselves

        self.atom_emb = AtomEmbeddingWrapper(
            atom_emb_dim, max_atomic_number + 1
        )  # +1 for absorbing state

        self.time_embedding = torch.nn.Linear(
            atom_emb_dim + t_emb_dim,
            atom_emb_dim,
        )

        self._set_adapter_embedding(condition_keys)

        if original_lattice_update:
            self.angle_edge_emb = nn.Sequential(
                nn.Linear(edge_emb_dim + 3, edge_emb_dim),
                nn.ReLU(),
                nn.Linear(edge_emb_dim, edge_emb_dim),
            )

            self.mlp_rbf_lattice = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.lattice_out_blocks = nn.ModuleList(
                [
                    RBFBasedLatticeUpdateBlockFrac(
                        edge_emb_dim,
                        activation,
                        emb_size_rbf,
                        edge_emb_dim,
                    )
                    for _ in range(num_blocks + 1)
                ]
            )

    def _set_adapter_embedding(self, condition_keys: list[str]):
        cond_adapters_per_layer = {}
        for cond_name in condition_keys:
            adapters = []

            for _ in range(self.config.num_blocks):
                adapters.append(ControlNetAdapter(self.config.atom_emb_dim))

            cond_adapters_per_layer[cond_name] = torch.nn.ModuleList(adapters)

        self.cond_adapters_per_layer = torch.nn.ModuleDict(cond_adapters_per_layer)

    @conditional_grad(torch.enable_grad())
    def forward(
        self,
        data,
        t_emb,
        added_cond=None,
        cond_mask=None,
        output_nodes_hidden_states: bool = False,
        output_edges_hidden_states: bool = False,
    ):
        data.build_graph(
            cutoff=self.config.cutoff, max_neighbors=self.config.max_neighbors
        )

        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data)
        distance_vec = -V_st * D_st[:, None]

        idx_s, idx_t = edge_index

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # Merge t_emb and atom embedding
        t_emb_per_atom = t_emb[batch]
        h = torch.cat([h, t_emb_per_atom], dim=1)
        h = self.time_embedding(h)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        if self.config.original_lattice_update:
            batch_edge = batch[edge_index[0]]
            cosines = torch.cosine_similarity(
                V_st[:, None], data.cell[batch_edge], dim=-1
            )
            m = torch.cat([m, cosines], dim=-1)
            m = self.angle_edge_emb(m)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        # (nAtoms, num_targets), (nEdges, num_targets)

        nodes_hidden_states = (h,) if output_nodes_hidden_states else None
        edges_hidden_states = (m,) if output_edges_hidden_states else None

        if added_cond is not None:
            added_cond_per_atom = {
                name: cond[batch] for name, cond in added_cond.items()
            }

            if cond_mask is not None:
                cond_mask_per_atom = {
                    name: cond[batch].float() for name, cond in cond_mask.items()
                }
            else:
                cond_mask_per_atom = {
                    name: torch.ones_like(cond) for name, cond in added_cond.items()
                }
        else:
            added_cond_per_atom = None
            cond_mask_per_atom = None

        if self.config.original_lattice_update:
            rbf_lattice = self.mlp_rbf_lattice(rbf)
            lattice_update = self.lattice_out_blocks[0](
                edge_emb=m,
                edge_index=edge_index,
                distance_vec=distance_vec,
                lattice=data.cell,
                batch=batch,
                rbf=rbf_lattice,
                normalize_score=True,
            )
        for i in range(self.num_blocks):
            if added_cond_per_atom is not None:
                h_adapt = torch.zeros_like(h)

                for cond_name, cond in added_cond_per_atom.items():
                    h_adapt_cond = self.cond_adapters_per_layer[cond_name][i](h, cond)

                    h_adapt += cond_mask_per_atom[cond_name] * h_adapt_cond

                h = h + h_adapt

            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_st += F
            E_t += E

            if self.config.original_lattice_update:
                lattice_update += self.lattice_out_blocks[i + 1](
                    edge_emb=m,
                    edge_index=edge_index,
                    distance_vec=distance_vec,
                    lattice=data.cell,
                    batch=batch,
                    rbf=rbf_lattice,
                    normalize_score=True,
                )

            if output_nodes_hidden_states:
                nodes_hidden_states += (h,)
            if output_edges_hidden_states:
                edges_hidden_states += (m,)

        nMolecules = torch.max(batch) + 1
        if self.extensive:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
            )  # (nMolecules, num_targets)

        outputs = {
            "energy": E_t,
            "nodes_last_hidden_state": h,
            "edges_last_hidden_state": m,
            "graph_edge_index": edge_index,
            "graph_edge_distance_vec": distance_vec,
            "graph_edge_distances": D_st,
        }

        if self.config.original_lattice_update:
            outputs["lattice_update"] = lattice_update

        if output_nodes_hidden_states:
            outputs["nodes_hidden_states"] = nodes_hidden_states

        if output_edges_hidden_states:
            outputs["edges_hidden_states"] = edges_hidden_states

        if self.regress_forces:
            if self.direct_forces:
                # map forces in edge directions
                F_st_vec = F_st[:, :, None] * V_st[:, None, :]
                # (nEdges, num_targets, 3)
                F_t = scatter(
                    F_st_vec,
                    idx_t,
                    dim=0,
                    dim_size=data.atomic_numbers.size(0),
                    reduce="add",
                )  # (nAtoms, num_targets, 3)
                F_t = F_t.squeeze(1)  # (nAtoms, 3)
            else:
                if self.num_targets > 1:
                    forces = []
                    for i in range(self.num_targets):
                        # maybe this can be solved differently
                        forces += [
                            -torch.autograd.grad(
                                E_t[:, i].sum(), pos, create_graph=True
                            )[0]
                        ]
                    F_t = torch.stack(forces, dim=1)
                    # (nAtoms, num_targets, 3)
                else:
                    F_t = -torch.autograd.grad(E_t.sum(), pos, create_graph=True)[0]
                    # (nAtoms, 3)

            outputs["forces"] = F_t

        return outputs

    # redefine function without safe_serialization
    # because shared weights are not supported
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "10GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=False,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )
