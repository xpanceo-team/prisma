from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from prisma.backbones.gemnet.layers.base_layers import ScaledSiLU

from diffusers.configuration_utils import register_to_config

from prisma.models.modeling_utils import ModelMixin
from prisma.configuration_utils import ConfigMixin
from prisma.data import StructureData
from prisma.utils.functions import cart_to_frac_coords_from_lattice_matrix

try:
    from prisma.models.gnns.mace import load_mace
except ImportError:
    load_mace = None


@dataclass
class MatterGenOutput:
    frac_coords_score: torch.tensor
    lattice_score: torch.tensor
    atom_types_logits: torch.tensor


class MatterGenModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        original_lattice_update: bool = False,
        max_atomic_number: int = 100,
        atom_emb_dim: int = 512,
        edge_emb_dim: int = 512,
        edge_emb_with_lattice_angles: bool = True,
        num_edge_blocks: int = 5,
    ):
        super().__init__()

        self.mlp_atom_types = nn.Linear(atom_emb_dim, max_atomic_number + 1)

        if not original_lattice_update:
            dim = edge_emb_dim + 3 if edge_emb_with_lattice_angles else edge_emb_dim
            self.mlp_edge_per_layer = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(dim, dim),
                        ScaledSiLU(),
                        torch.nn.Linear(dim, 1),
                    )
                    for _ in range(num_edge_blocks)
                ]
            )

    def forward(self, data: StructureData, gnn_output: dict):
        atom_types_logits = self.mlp_atom_types(gnn_output["nodes_last_hidden_state"])

        cart_coords_score = gnn_output["forces"]
        frac_coords_score = cart_to_frac_coords_from_lattice_matrix(
            cart_coords_score, data.cell, data.num_atoms
        )

        # here maybe we should use output.edges_hidden_states[1:] and not include embedding layer
        edge_index_batch = data.batch[gnn_output["graph_edge_index"][0]]

        if self.config.original_lattice_update:
            lattice_score = gnn_output["lattice_update"]
        else:
            edge_score = self._get_edge_scores(
                edges_hidden_states=gnn_output["edges_hidden_states"],
                edge_index_batch=edge_index_batch,
                lattice=data.cell,
                graph_edge_distance_vec=gnn_output["graph_edge_distance_vec"],
                graph_edge_distances=gnn_output["graph_edge_distances"],
            )

            lattice_score = self._get_lattice_score(
                edge_index_batch=edge_index_batch,
                graph_edge_distance_vec=gnn_output["graph_edge_distance_vec"],
                graph_edge_distances=gnn_output["graph_edge_distances"],
                edge_score=edge_score,
            )

        output = MatterGenOutput(
            frac_coords_score=frac_coords_score,
            lattice_score=lattice_score,
            atom_types_logits=atom_types_logits,
        )

        return output

    def _get_edge_scores(
        self,
        edges_hidden_states,
        edge_index_batch,
        lattice,
        graph_edge_distance_vec,
        graph_edge_distances,
    ):
        layer_outputs = []
        for mlp_idx, mlp in enumerate(self.mlp_edge_per_layer):
            edges_hidden_state = edges_hidden_states[mlp_idx]
            edge_repr = self._get_edge_representation(
                edges_hidden_state=edges_hidden_state,
                edge_index_batch=edge_index_batch,
                lattice=lattice,
                graph_edge_distance_vec=graph_edge_distance_vec,
                graph_edge_distances=graph_edge_distances,
            )

            layer_outputs.append(mlp(edge_repr))

        edge_scores = torch.stack(layer_outputs).squeeze(-1)

        return edge_scores

    def _get_edge_representation(
        self,
        edges_hidden_state,
        edge_index_batch,
        lattice,
        graph_edge_distance_vec,
        graph_edge_distances,
    ):
        if not self.config.edge_emb_with_lattice_angles:
            return edges_hidden_state

        lattice_broadcast = lattice[edge_index_batch, :, :]
        lengths_broadcast = torch.norm(lattice_broadcast, p=2, dim=1)

        dot_product = (graph_edge_distance_vec.unsqueeze(2) * lattice_broadcast).sum(1)
        cosines = dot_product / lengths_broadcast / graph_edge_distances.unsqueeze(1)

        edges_hidden_state = torch.cat([edges_hidden_state, cosines], dim=1)

        return edges_hidden_state

    def _get_lattice_score(
        self,
        edge_index_batch,
        graph_edge_distance_vec,
        graph_edge_distances,
        edge_score,
    ):
        num_edges = edge_index_batch.unique(return_counts=True)[1]

        num_edges_broadcast = torch.repeat_interleave(num_edges, num_edges)

        # [num_layers, edges] = [num_layers, edges] * [1, edges] / [1, edges]
        phi_array = (
            edge_score
            / graph_edge_distances[None, :] ** 2
            / num_edges_broadcast[None, :]
        )

        # [num_layers, edges, 3] = [num_layers, edges, 1] * [1, edges, 3]
        distance_vec_scaled = phi_array[..., None] * graph_edge_distance_vec[None, :]

        # [num_layers, 3, edges, 3] = [1, 3, edges, 1] * [num_layers, 1, edges, 3]
        outer_product = (
            graph_edge_distance_vec.T[None, ..., None]
            * distance_vec_scaled[:, None, ...]
        )

        # [num_layers, 3, batch_size, 3]
        result = scatter_add(outer_product, edge_index_batch, dim=2)

        # [3, batch_size, 3]
        result = result.sum(0)

        # [batch_size, 3, 3]
        result = result.transpose(0, 1)

        return result
