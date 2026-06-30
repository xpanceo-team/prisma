from __future__ import annotations

from typing import Dict

import os

import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_add
from fairchem.core.models.gemnet.layers.base_layers import Dense

from diffusers.configuration_utils import register_to_config

from crystal_diffusers.configuration_utils import ConfigMixin
from crystal_diffusers.models.embeddings import ControlNetAdapter
from crystal_diffusers.models.modeling_utils import ModelMixin

from metatensor.torch import Labels, TensorBlock
from metatomic.torch import System, register_autograd_neighbors
from metatrain.pet import PET
from metatrain.pet.modules.structures import (
    concatenate_structures,
    compute_reversed_neighbor_list,
    cutoff_func_bump,
    cutoff_func_cosine,
    edge_array_to_nef,
    get_adaptive_cutoffs,
    get_corresponding_edges,
    get_nef_indices,
)
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info

PET_HARD_MAX_NEIGHBORS = 50.0
PET_GRAPH_MAX_NEIGHBORS = 256.0


class GaussianRBF(nn.Module):
    def __init__(self, num_rbfs: int = 32, cutoff: float = 6.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbfs)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0), requires_grad=False)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances[:, None] - self.centers[None, :]
        return torch.exp(-self.gamma * diff**2)


def edge_score_to_lattice_score_frac_symmetric(
    score_d: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    batch_edge = batch[edge_index[0]]
    unit_edge_vectors = edge_vectors / edge_vectors.norm(dim=-1, keepdim=True).clamp(
        min=1e-8
    )
    score_lattice = scatter_add(
        score_d[:, None, None]
        * (unit_edge_vectors[:, :, None] @ unit_edge_vectors[:, None, :]),
        batch_edge,
        dim=0,
        dim_size=batch.max() + 1,
    ).transpose(-1, -2)
    return score_lattice


class PETLatticeUpdateBlockFrac(nn.Module):
    def __init__(
        self,
        edge_emb_dim: int,
        num_rbfs: int,
        activation: str = "swish",
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.edge_mlp = nn.Sequential(
            Dense(edge_emb_dim, edge_emb_dim, activation=activation),
            Dense(edge_emb_dim, edge_emb_dim),
        )
        self.rbf_proj = Dense(num_rbfs, edge_emb_dim, activation=None, bias=False)
        self.out = Dense(edge_emb_dim, num_heads, bias=False, activation=None)

    def forward(
        self,
        edge_emb: torch.Tensor,
        edge_index: torch.Tensor,
        distance_vec: torch.Tensor,
        batch: torch.Tensor,
        rbf: torch.Tensor,
        normalize_score: bool = True,
    ) -> torch.Tensor:
        x = self.edge_mlp(edge_emb)
        x = x * self.rbf_proj(rbf)
        edge_scores = self.out(x)

        if normalize_score:
            num_edges = scatter(torch.ones_like(distance_vec[:, 0]), batch[edge_index[0]])
            edge_scores = edge_scores / num_edges[batch[edge_index[0]], None].clamp(min=1.0)

        outs = []
        for i in range(self.num_heads):
            outs.append(
                edge_score_to_lattice_score_frac_symmetric(
                    score_d=edge_scores[:, i],
                    edge_index=edge_index,
                    edge_vectors=distance_vec,
                    batch=batch,
                )
            )
        return torch.stack(outs, dim=-1).sum(-1)


class PETForceReadoutBlock(nn.Module):
    """GemNet-style per-block force readout with RBF gating."""

    def __init__(
        self,
        edge_emb_dim: int,
        num_rbfs: int,
        activation: str = "swish",
    ):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            Dense(edge_emb_dim, edge_emb_dim, activation=activation),
            Dense(edge_emb_dim, edge_emb_dim),
        )
        self.rbf_proj = Dense(num_rbfs, edge_emb_dim, activation=None, bias=False)
        self.out = Dense(edge_emb_dim, 1, bias=False, activation=None)

    def forward(self, edge_emb: torch.Tensor, rbf: torch.Tensor) -> torch.Tensor:
        x = self.edge_mlp(edge_emb)
        x = x * self.rbf_proj(rbf)
        return self.out(x).squeeze(-1)


class PETWrapper(PET, ModelMixin, ConfigMixin):
    """MatterGen-compatible PET wrapper with PET-conditioned backbone and PET edge readouts."""

    @register_to_config
    def __init__(
        self,
        t_emb_dim: int = 512,
        condition_keys: list[str] | None = None,
        condition_dim: int | None = None,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        graph_max_neighbors: int = 256,
        use_adaptive_neighbors: bool = True,
        max_atomic_number: int = 100,
        atom_emb_dim: int = 512,
        edge_emb_dim: int = 512,
        head_dim: int = 128,
        feedforward_dim: int = 256,
        num_attention_layers: int = 2,
        featurizer_type: str = "feedforward",
        num_blocks: int = 4,
        extensive: bool = False,
        original_lattice_update: bool = False,
        num_rbfs: int = 32,
        readout_activation: str = "swish",
    ):
        if condition_keys is None:
            condition_keys = []
        if condition_dim is None:
            condition_dim = t_emb_dim

        pet_hypers = get_default_hypers("pet")["model"]
        pet_hypers["cutoff"] = float(cutoff)
        pet_hypers["num_gnn_layers"] = int(num_blocks)
        pet_hypers["featurizer_type"] = featurizer_type
        pet_hypers["d_node"] = int(atom_emb_dim)
        pet_hypers["d_pet"] = int(edge_emb_dim)
        pet_hypers["d_head"] = int(head_dim)
        pet_hypers["d_feedforward"] = int(feedforward_dim)
        pet_hypers["num_attention_layers"] = int(num_attention_layers)
        # Keep an internal hard neighbor cap for memory/runtime parity with GemNet.
        pet_hypers["num_neighbors_adaptive"] = (
            float(max_neighbors) if use_adaptive_neighbors else None
        )

        dataset_info = DatasetInfo(
            length_unit="Angstrom",
            atomic_types=list(range(max_atomic_number + 1)),
            targets={
                "energy": get_energy_target_info(
                    "energy", {"quantity": "energy", "unit": "eV"}
                )
            },
        )

        super().__init__(hypers=pet_hypers, dataset_info=dataset_info)

        self.t_in_projector = nn.Linear(t_emb_dim, self.d_node)
        self.time_embedding_out = nn.Linear(atom_emb_dim + t_emb_dim, atom_emb_dim)
        self.node_energy_head = nn.Linear(atom_emb_dim, 1)
        self.edge_energy_head = nn.Linear(edge_emb_dim, 1)

        self.rbf = GaussianRBF(num_rbfs=num_rbfs, cutoff=cutoff)
        self.angle_edge_emb = nn.Sequential(
            Dense(edge_emb_dim + 3, edge_emb_dim, activation=readout_activation),
            Dense(edge_emb_dim, edge_emb_dim),
        )
        self.force_out_blocks = nn.ModuleList(
            [
                PETForceReadoutBlock(
                    edge_emb_dim=edge_emb_dim,
                    num_rbfs=num_rbfs,
                    activation=readout_activation,
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.lattice_out_blocks = nn.ModuleList(
            [
                PETLatticeUpdateBlockFrac(
                    edge_emb_dim=edge_emb_dim,
                    num_rbfs=num_rbfs,
                    activation=readout_activation,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self._set_adapter_embedding(condition_keys, int(condition_dim))

    def _set_adapter_embedding(self, condition_keys: list[str], condition_dim: int):
        cond_projectors = {}
        cond_adapters_per_layer = {}
        for cond_name in condition_keys:
            cond_projectors[cond_name] = nn.Linear(condition_dim, self.d_node)
            cond_adapters_per_layer[cond_name] = nn.ModuleList(
                [ControlNetAdapter(self.d_node) for _ in range(self.config.num_blocks)]
            )

        self.cond_projectors = nn.ModuleDict(cond_projectors)
        self.cond_adapters_per_layer = nn.ModuleDict(cond_adapters_per_layer)

    def _resolve_neighbor_cap(self) -> int:
        cap = self.config.max_neighbors
        if cap is None:
            cap = PET_HARD_MAX_NEIGHBORS
        try:
            cap = int(float(cap))
        except (TypeError, ValueError):
            cap = int(PET_HARD_MAX_NEIGHBORS)
        return max(cap, 1)

    def _resolve_graph_neighbor_cap(self) -> int:
        cap = self.config.graph_max_neighbors
        if cap is None:
            cap = max(PET_GRAPH_MAX_NEIGHBORS, self._resolve_neighbor_cap())
        try:
            cap = int(float(cap))
        except (TypeError, ValueError):
            cap = int(PET_GRAPH_MAX_NEIGHBORS)
        return max(cap, self._resolve_neighbor_cap(), 1)

    def _build_neighbor_list_tensor(
        self,
        *,
        positions: torch.Tensor,
        cell: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        cell_shifts: torch.Tensor,
    ) -> TensorBlock:
        cell_shifts_int = torch.round(cell_shifts).to(
            device=centers.device, dtype=torch.int32
        )
        edge_vectors = (
            positions[neighbors]
            - positions[centers]
            + cell_shifts_int.to(cell.dtype) @ cell
        )
        samples = torch.cat(
            [
                centers[:, None].to(torch.int32),
                neighbors[:, None].to(torch.int32),
                cell_shifts_int,
            ],
            dim=1,
        )
        samples_labels = Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=samples,
            assume_unique=True,
        ).to(device=edge_vectors.device)
        components = [Labels.range("xyz", 3).to(device=edge_vectors.device)]
        properties = Labels.range("distance", 1).to(device=edge_vectors.device)
        neighbors_block = TensorBlock(
            values=edge_vectors.unsqueeze(-1),
            samples=samples_labels,
            components=components,
            properties=properties,
        )
        return neighbors_block.to(device=positions.device, dtype=positions.dtype)

    def _to_metatomic_systems(self, data) -> list[System]:
        # Avoid metatrain/vesin neighbor-list hangs by reusing the project's
        # PBC graph builder. Keep a larger candidate buffer here so PET's own
        # adaptive neighbor pruning still sees enough local environment context.
        data.build_graph(
            cutoff=float(self.cutoff),
            max_neighbors=(
                self._resolve_graph_neighbor_cap()
                if self.config.use_adaptive_neighbors
                else self._resolve_neighbor_cap()
            ),
        )
        requested_options = self.requested_neighbor_lists()[0]

        systems = []
        atom_offset = 0
        for i, n_atoms in enumerate(data.num_atoms.tolist()):
            atom_slice = slice(atom_offset, atom_offset + n_atoms)
            structure_edge_mask = (
                (data.edge_index[1] >= atom_offset)
                & (data.edge_index[1] < atom_offset + n_atoms)
                & (data.edge_index[0] >= atom_offset)
                & (data.edge_index[0] < atom_offset + n_atoms)
            )
            centers = data.edge_index[1, structure_edge_mask] - atom_offset
            neighbors = data.edge_index[0, structure_edge_mask] - atom_offset
            cell_shifts = data.cell_offsets[structure_edge_mask]
            positions = data.pos[atom_slice]

            if hasattr(data, "pbc"):
                pbc = data.pbc[i].to(data.pos.device)
            else:
                # crystal workflow default
                pbc = torch.tensor([True, True, True], device=data.pos.device)
            system = System(
                types=data.atomic_numbers[atom_slice].long(),
                positions=data.pos[atom_slice],
                cell=data.cell[i],
                pbc=pbc,
            )
            neighbors_block = self._build_neighbor_list_tensor(
                positions=positions,
                cell=data.cell[i],
                centers=centers.long(),
                neighbors=neighbors.long(),
                cell_shifts=cell_shifts,
            )
            register_autograd_neighbors(system, neighbors_block)
            system.add_neighbor_list(requested_options, neighbors_block)
            systems.append(system)
            atom_offset += n_atoms
        return systems

    def _get_pet_to_pyg_index(
        self,
        sample_labels,
        num_atoms: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        offsets = torch.cumsum(
            torch.cat([torch.tensor([0], device=device, dtype=torch.long), num_atoms[:-1].to(device)]),
            dim=0,
        )
        sample_vals = sample_labels.values.to(device=device, dtype=torch.long)
        return offsets[sample_vals[:, 0]] + sample_vals[:, 1]

    def _map_structure_cond_to_pet_order(
        self,
        cond_per_structure: torch.Tensor,
        sample_labels,
        num_atoms: torch.Tensor,
        device: torch.device,
    ):
        cond_per_atom = cond_per_structure.to(device).repeat_interleave(num_atoms.to(device), dim=0)
        pet_to_pyg = self._get_pet_to_pyg_index(sample_labels, num_atoms, device)
        return cond_per_atom[pet_to_pyg]

    def _prepare_pet_graph(self, data):
        systems = self._to_metatomic_systems(data)
        options = self.requested_neighbor_lists()[0]
        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
            system_indices,
            sample_labels,
        ) = concatenate_structures(systems, options)

        if len(cells) == 1:
            cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
        else:
            cell_contributions = torch.einsum(
                "ab, abc -> ac", cell_shifts.to(cells.dtype), cells[system_indices[centers]]
            )

        num_nodes = positions.shape[0]
        edge_vec_pet = positions[neighbors] - positions[centers] + cell_contributions
        edge_dist_pet = torch.norm(edge_vec_pet, dim=-1) + 1e-15

        if self.num_neighbors_adaptive is not None:
            atomic_cutoffs = get_adaptive_cutoffs(
                centers,
                edge_dist_pet,
                self.num_neighbors_adaptive,
                num_nodes,
                options.cutoff,
                cutoff_width=self.cutoff_width,
            )
            pair_cutoffs = (atomic_cutoffs[centers] + atomic_cutoffs[neighbors]) / 2.0
            cutoff_mask = edge_dist_pet <= pair_cutoffs

            centers = centers[cutoff_mask]
            neighbors = neighbors[cutoff_mask]
            edge_vec_pet = edge_vec_pet[cutoff_mask]
            edge_dist_pet = edge_dist_pet[cutoff_mask]
            cell_shifts = cell_shifts[cutoff_mask]
            pair_cutoffs = pair_cutoffs[cutoff_mask]
        else:
            pair_cutoffs = options.cutoff * torch.ones(
                len(centers),
                device=positions.device,
                dtype=positions.dtype,
            )

        num_neighbors = torch.bincount(centers, minlength=num_nodes)
        max_edges_per_node = (
            int(torch.max(num_neighbors)) if num_neighbors.numel() > 0 else 0
        )

        if self.cutoff_function.lower() == "bump":
            cutoff_factors_flat = cutoff_func_bump(
                edge_dist_pet, pair_cutoffs, self.cutoff_width
            )
        elif self.cutoff_function.lower() == "cosine":
            cutoff_factors_flat = cutoff_func_cosine(
                edge_dist_pet, pair_cutoffs, self.cutoff_width
            )
        else:
            raise ValueError(
                f"Unknown cutoff function type: {self.cutoff_function}. "
                "Supported types are 'Cosine' and 'Bump'."
            )

        nef_indices, _, padding_mask = get_nef_indices(
            centers, num_nodes, max_edges_per_node
        )
        element_indices_nodes = self.species_to_species_index[species]
        element_indices_neighbors = element_indices_nodes[neighbors]

        edge_vectors_nef = edge_array_to_nef(edge_vec_pet, nef_indices)
        edge_distances_nef = torch.sqrt(torch.sum(edge_vectors_nef**2, dim=2) + 1e-15)
        element_indices_neighbors = edge_array_to_nef(
            element_indices_neighbors, nef_indices
        )
        cutoff_factors_nef = edge_array_to_nef(
            cutoff_factors_flat, nef_indices, padding_mask, 0.0
        )

        corresponding_edges = get_corresponding_edges(
            torch.concatenate(
                [centers.unsqueeze(-1), neighbors.unsqueeze(-1), cell_shifts], dim=-1
            )
        )
        reversed_neighbor_list = compute_reversed_neighbor_list(
            nef_indices, corresponding_edges, padding_mask
        )
        neighbors_index = edge_array_to_nef(neighbors, nef_indices).to(torch.int64)
        reverse_neighbor_index = (
            neighbors_index * neighbors_index.shape[1] + reversed_neighbor_list
        )
        reverse_neighbor_index[~padding_mask] = torch.arange(
            int(torch.sum(~padding_mask)), device=reverse_neighbor_index.device
        )

        num_edges = centers.shape[0]

        return {
            "systems": systems,
            "element_indices_nodes": element_indices_nodes,
            "element_indices_neighbors": element_indices_neighbors,
            "edge_vectors_nef": edge_vectors_nef,
            "edge_distances_nef": edge_distances_nef,
            "padding_mask": padding_mask,
            "reverse_neighbor_index": reverse_neighbor_index,
            "cutoff_factors_nef": cutoff_factors_nef,
            "sample_labels": sample_labels,
            "centers": centers,
            "neighbors": neighbors,
            "edge_vec_pet": edge_vec_pet,
            "edge_dist_pet": edge_dist_pet,
            "nef_indices": nef_indices,
            "num_edges": num_edges,
        }

    def _nef_to_flat_edges(self, nef_tensor: torch.Tensor, nef_indices: torch.Tensor, padding_mask: torch.Tensor, num_edges: int):
        out = torch.zeros((num_edges, nef_tensor.shape[-1]), device=nef_tensor.device, dtype=nef_tensor.dtype)
        out[nef_indices[padding_mask].long()] = nef_tensor[padding_mask]
        return out

    def _debug_check_edge_alignment(self, graph):
        # Set CRYSTAL_DIFFUSERS_DEBUG_PET_ALIGN=1 to enable this runtime check.
        if os.getenv("CRYSTAL_DIFFUSERS_DEBUG_PET_ALIGN", "0") != "1":
            return

        edge_vec_flat = self._nef_to_flat_edges(
            graph["edge_vectors_nef"],
            graph["nef_indices"],
            graph["padding_mask"],
            graph["num_edges"],
        )
        edge_dist_flat = torch.zeros(
            graph["num_edges"],
            device=graph["edge_distances_nef"].device,
            dtype=graph["edge_distances_nef"].dtype,
        )
        edge_dist_flat[graph["nef_indices"][graph["padding_mask"]].long()] = graph[
            "edge_distances_nef"
        ][graph["padding_mask"]]

        vec_ref = graph["edge_vec_pet"].to(edge_vec_flat.device)
        dist_ref = graph["edge_dist_pet"].to(edge_dist_flat.device)

        if not torch.allclose(edge_vec_flat, vec_ref, atol=1e-6, rtol=1e-5) or not torch.allclose(
            edge_dist_flat, dist_ref, atol=1e-6, rtol=1e-5
        ):
            raise RuntimeError(
                "PETWrapper: NEF→flat edge alignment mismatch; edge embeddings/cutoffs likely scrambled."
            )

    def _run_pet_backbone(self, graph, data, t_emb, added_cond=None, cond_mask=None):
        device = data.pos.device
        use_manual_attention = graph["edge_vectors_nef"].requires_grad and self.training

        backbone_cond_per_atom = None
        backbone_mask_per_atom = None
        if added_cond is not None:
            backbone_cond_per_atom = {}
            backbone_mask_per_atom = {}
            for cond_name, cond in added_cond.items():
                backbone_cond_per_atom[cond_name] = self._map_structure_cond_to_pet_order(
                    cond, graph["sample_labels"], data.num_atoms, device
                )
            if cond_mask is not None:
                for cond_name, mask in cond_mask.items():
                    backbone_mask_per_atom[cond_name] = self._map_structure_cond_to_pet_order(
                        mask.float(), graph["sample_labels"], data.num_atoms, device
                    )
            else:
                for cond_name, cond in backbone_cond_per_atom.items():
                    backbone_mask_per_atom[cond_name] = torch.ones(
                        cond.shape[0],
                        1,
                        device=cond.device,
                        dtype=cond.dtype,
                    )

        # timestep conditioning in PET order, injected before every PET GNN layer
        t_pyg = t_emb[data.batch]
        pet_to_pyg = self._get_pet_to_pyg_index(graph["sample_labels"], data.num_atoms, device)
        t_pet = t_pyg[pet_to_pyg]
        t_node = self.t_in_projector(t_pet)

        edge_states_nef = []
        node_states = []

        input_edge_embeddings = self.edge_embedder(graph["element_indices_neighbors"])
        edge_states_nef.append(input_edge_embeddings)

        if self.featurizer_type == "feedforward":
            input_node_embeddings = self.node_embedders[0](graph["element_indices_nodes"])
            for i, (combination_norm, combination_mlp, gnn_layer) in enumerate(
                zip(
                    self.combination_norms,
                    self.combination_mlps,
                    self.gnn_layers,
                    strict=True,
                )
            ):
                layer_node_embeddings = input_node_embeddings + t_node

                if backbone_cond_per_atom is not None and self.cond_adapters_per_layer:
                    h_adapt = torch.zeros_like(layer_node_embeddings)
                    for cond_name, cond in backbone_cond_per_atom.items():
                        cond = self.cond_projectors[cond_name](cond)
                        h_adapt_cond = self.cond_adapters_per_layer[cond_name][i](
                            layer_node_embeddings, cond
                        )
                        h_adapt += backbone_mask_per_atom[cond_name].float() * h_adapt_cond
                    layer_node_embeddings = layer_node_embeddings + h_adapt

                output_node_embeddings, output_edge_embeddings = gnn_layer(
                    layer_node_embeddings,
                    input_edge_embeddings,
                    graph["element_indices_neighbors"],
                    graph["edge_vectors_nef"],
                    graph["padding_mask"],
                    graph["edge_distances_nef"],
                    graph["cutoff_factors_nef"],
                    use_manual_attention,
                )
                node_states.append(output_node_embeddings)
                edge_states_nef.append(output_edge_embeddings)

                input_node_embeddings = output_node_embeddings
                new_input_edge_embeddings = output_edge_embeddings.reshape(
                    output_edge_embeddings.shape[0] * output_edge_embeddings.shape[1],
                    output_edge_embeddings.shape[2],
                )[graph["reverse_neighbor_index"]].reshape(
                    output_edge_embeddings.shape[0],
                    output_edge_embeddings.shape[1],
                    output_edge_embeddings.shape[2],
                )
                concatenated = torch.cat(
                    [output_edge_embeddings, new_input_edge_embeddings], dim=-1
                )
                input_edge_embeddings = (
                    input_edge_embeddings
                    + output_edge_embeddings
                    + combination_mlp(combination_norm(concatenated))
                )
        else:
            for i, (node_embedder, gnn_layer) in enumerate(
                zip(self.node_embedders, self.gnn_layers, strict=True)
            ):
                input_node_embeddings = node_embedder(graph["element_indices_nodes"])
                input_node_embeddings = input_node_embeddings + t_node

                if backbone_cond_per_atom is not None and self.cond_adapters_per_layer:
                    h_adapt = torch.zeros_like(input_node_embeddings)
                    for cond_name, cond in backbone_cond_per_atom.items():
                        cond = self.cond_projectors[cond_name](cond)
                        h_adapt_cond = self.cond_adapters_per_layer[cond_name][i](
                            input_node_embeddings, cond
                        )
                        h_adapt += backbone_mask_per_atom[cond_name].float() * h_adapt_cond
                    input_node_embeddings = input_node_embeddings + h_adapt

                output_node_embeddings, output_edge_embeddings = gnn_layer(
                    input_node_embeddings,
                    input_edge_embeddings,
                    graph["element_indices_neighbors"],
                    graph["edge_vectors_nef"],
                    graph["padding_mask"],
                    graph["edge_distances_nef"],
                    graph["cutoff_factors_nef"],
                    use_manual_attention,
                )
                node_states.append(output_node_embeddings)
                edge_states_nef.append(output_edge_embeddings)

                new_input_messages = output_edge_embeddings.reshape(
                    output_edge_embeddings.shape[0] * output_edge_embeddings.shape[1],
                    output_edge_embeddings.shape[2],
                )[graph["reverse_neighbor_index"]].reshape(
                    output_edge_embeddings.shape[0],
                    output_edge_embeddings.shape[1],
                    output_edge_embeddings.shape[2],
                )
                input_edge_embeddings = 0.5 * (input_edge_embeddings + new_input_messages)

        if self.long_range:
            long_range_features = self._calculate_long_range_features(
                graph["systems"],
                node_states,
                graph["edge_distances_nef"],
                graph["padding_mask"],
            )
            for i in range(len(node_states)):
                node_states[i] = (node_states[i] + long_range_features) * (0.5**0.5)

        pet_to_pyg = self._get_pet_to_pyg_index(graph["sample_labels"], data.num_atoms, device)

        node_states_pyg = []
        for node_state in node_states:
            out = torch.zeros_like(node_state)
            out[pet_to_pyg] = node_state
            node_states_pyg.append(out)

        edge_states_flat = [
            self._nef_to_flat_edges(es, graph["nef_indices"], graph["padding_mask"], graph["num_edges"])
            for es in edge_states_nef
        ]

        cutoff_flat = torch.zeros(
            graph["num_edges"],
            device=graph["cutoff_factors_nef"].device,
            dtype=graph["cutoff_factors_nef"].dtype,
        )
        cutoff_flat[graph["nef_indices"][graph["padding_mask"]].long()] = graph["cutoff_factors_nef"][graph["padding_mask"]]

        return node_states_pyg, edge_states_flat, cutoff_flat

    def forward(
        self,
        data,
        t_emb,
        added_cond=None,
        cond_mask=None,
        output_nodes_hidden_states: bool = False,
        output_edges_hidden_states: bool = False,
    ):
        batch = data.batch
        graph = self._prepare_pet_graph(data)
        self._debug_check_edge_alignment(graph)

        pet_to_pyg = self._get_pet_to_pyg_index(
            graph["sample_labels"], data.num_atoms, data.pos.device
        ).to(device=data.pos.device, dtype=torch.long)
        edge_index_pet = torch.stack([graph["centers"].long(), graph["neighbors"].long()], dim=0)
        edge_index = pet_to_pyg[edge_index_pet].to(device=data.pos.device, dtype=torch.long)
        edge_vec = graph["edge_vec_pet"].to(data.pos.device)
        edge_dist = graph["edge_dist_pet"].to(data.pos.device)

        node_states_pyg, edge_states_flat, cutoff_flat = self._run_pet_backbone(
            graph,
            data,
            t_emb=t_emb,
            added_cond=added_cond,
            cond_mask=cond_mask,
        )

        pet_h = node_states_pyg[-1]
        t_emb_per_atom = t_emb[batch]
        h = self.time_embedding_out(torch.cat([pet_h, t_emb_per_atom], dim=-1))
        num_nodes = h.size(0)

        if edge_index.numel() == 0:
            raise RuntimeError(
                "PETWrapper: graph has no edges inside the cutoff. "
                f"cutoff={float(self.cutoff):.3f}, "
                f"max_neighbors={self._resolve_neighbor_cap()}, "
                f"num_atoms={data.num_atoms.tolist()}"
            )

        min_edge_idx = int(edge_index.min().item())
        max_edge_idx = int(edge_index.max().item())
        if min_edge_idx < 0 or max_edge_idx >= num_nodes:
            raise RuntimeError(
                "PETWrapper: edge_index is out of node bounds "
                f"(min={min_edge_idx}, max={max_edge_idx}, num_nodes={num_nodes})."
            )

        if not torch.isfinite(edge_vec).all() or not torch.isfinite(edge_dist).all():
            raise RuntimeError("PETWrapper: non-finite edge geometry detected.")

        rbf = self.rbf(edge_dist)
        edge_states_for_readout = edge_states_flat
        if self.config.original_lattice_update:
            batch_edge = batch[edge_index[0]]
            edge_unit_vec = edge_vec / edge_dist[:, None].clamp(min=1e-8)
            cosines = torch.cosine_similarity(
                edge_unit_vec[:, None],
                data.cell[batch_edge],
                dim=-1,
            )
            edge_states_for_readout = tuple(
                self.angle_edge_emb(torch.cat([edge_state, cosines], dim=-1))
                for edge_state in edge_states_flat
            )

        node_energy = self.node_energy_head(h).squeeze(-1)
        edge_energy = self.edge_energy_head(edge_states_flat[-1]).squeeze(-1) * cutoff_flat
        # Edge contribution is accumulated to edge centers (src / edge_index[0]).
        edge_to_node = torch.zeros(
            num_nodes,
            device=edge_energy.device,
            dtype=edge_energy.dtype,
        )
        edge_to_node.index_add_(0, edge_index[0], edge_energy)
        num_structures = int(data.num_atoms.shape[0])
        total_energy = torch.zeros(
            num_structures,
            device=node_energy.device,
            dtype=node_energy.dtype,
        )
        total_energy.index_add_(0, batch, node_energy + edge_to_node)
        if not self.config.extensive:
            num_atoms = data.num_atoms.to(
                device=total_energy.device,
                dtype=total_energy.dtype,
            )
            total_energy = total_energy / num_atoms.clamp(min=1.0)
        total_energy = total_energy.unsqueeze(-1)

        # Predict scalar edge contributions and project to edge directions
        # to preserve rotational equivariance of force outputs.
        # Aggregate force readout over all PET edge states (input + each block),
        # analogous to GemNet's cumulative force contributions.
        edge_force = torch.zeros_like(cutoff_flat)
        for force_block, edge_state in zip(
            self.force_out_blocks, edge_states_for_readout, strict=True
        ):
            edge_force = edge_force + force_block(edge_state, rbf)
        edge_force = edge_force * cutoff_flat
        edge_unit_vec = edge_vec / edge_dist[:, None].clamp(min=1e-8)
        edge_force_vec = edge_force[:, None] * edge_unit_vec
        # Accumulate on destination nodes to match GemNet force assignment semantics.
        forces = torch.zeros(
            (num_nodes, edge_force_vec.shape[-1]),
            device=edge_force_vec.device,
            dtype=edge_force_vec.dtype,
        )
        forces.index_add_(0, edge_index[1], edge_force_vec)

        lattice_update = None
        if self.config.original_lattice_update:
            lattice_update = edge_vec.new_zeros((data.num_atoms.shape[0], 3, 3))
            for lattice_block, edge_state in zip(
                self.lattice_out_blocks, edge_states_for_readout, strict=True
            ):
                layer_lattice_update = lattice_block(
                    edge_emb=edge_state,
                    edge_index=edge_index,
                    distance_vec=edge_vec,
                    batch=batch,
                    rbf=rbf,
                    normalize_score=True,
                )
                lattice_update = lattice_update + layer_lattice_update

        outputs: Dict[str, torch.Tensor] = {
            "energy": total_energy,
            "forces": forces,
            "nodes_last_hidden_state": h,
            "edges_last_hidden_state": edge_states_flat[-1],
            "graph_edge_index": edge_index,
            "graph_edge_distance_vec": edge_vec,
            "graph_edge_distances": edge_dist,
        }

        if self.config.original_lattice_update:
            outputs["lattice_update"] = lattice_update

        if output_nodes_hidden_states:
            outputs["nodes_hidden_states"] = tuple(node_states_pyg) + (h,)
        if output_edges_hidden_states:
            outputs["edges_hidden_states"] = tuple(edge_states_flat)

        return outputs


# Backward-compatible alias for historical checkpoints/config targets.
PETMADWrapper = PETWrapper
