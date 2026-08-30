from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_add

from diffusers.models.modeling_utils import get_parameter_device, get_parameter_dtype
from prisma.backbones.common import gp_utils
from prisma.backbones.common.utils import conditional_grad
from prisma.backbones.equiformer_v2.equiformer_v2 import (
    EquiformerV2Backbone,
)
from prisma.backbones.equiformer_v2.heads.scalar import (
    EqV2ScalarHead,
)
from prisma.backbones.equiformer_v2.heads.vector import (
    EqV2VectorHead,
)
from prisma.backbones.equiformer_v2.radial_function import RadialFunction
from prisma.backbones.equiformer_v2.so3 import SO3_Embedding

from diffusers.configuration_utils import register_to_config

from prisma.configuration_utils import ConfigMixin
from prisma.models.embeddings import ControlNetAdapter
from prisma.models.modeling_utils import ModelMixin


def rename_keys(state_dict):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        prefix = "module.module."
        new_key = key[len(prefix) :] if key.startswith(prefix) else key
        new_state_dict[new_key] = value

    return new_state_dict


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


class EquiformerLatticeUpdateBlockFrac(nn.Module):
    def __init__(self, edge_emb_dim: int, num_rbfs: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_emb_dim, edge_emb_dim),
            nn.SiLU(),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )
        self.rbf_proj = nn.Linear(num_rbfs, edge_emb_dim, bias=False)
        self.out = nn.Linear(edge_emb_dim, num_heads, bias=False)

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
            edge_scores = edge_scores / num_edges[batch[edge_index[0]], None].clamp(
                min=1.0
            )

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


def load_equiformer_v2(
    checkpoint_path: Optional[str] = None,
    load_states: bool = True,
    model_config: Optional[dict] = None,
    **kwargs,
):
    config = {}
    if checkpoint_path is not None and load_states:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config.update(checkpoint["config"]["model_attributes"])

    if model_config is not None:
        config.update(model_config)

    config.update(kwargs)
    config["checkpoint_path"] = checkpoint_path
    config["load_states"] = load_states

    model = EquiformerV2Wrapper(**config)
    return model, config


class EquiformerV2Wrapper(EquiformerV2Backbone, ModelMixin, ConfigMixin):
    config_name = "config.json"

    @property
    def device(self):
        try:
            return get_parameter_device(self)
        except StopIteration:
            return torch.device(getattr(self, "_device_override", "cpu"))

    @device.setter
    def device(self, value):
        self._device_override = value

    @property
    def dtype(self):
        try:
            return get_parameter_dtype(self)
        except StopIteration:
            return getattr(self, "_dtype_override", torch.get_default_dtype())

    @dtype.setter
    def dtype(self, value):
        self._dtype_override = value

    @register_to_config
    def __init__(
        self,
        checkpoint_path: str | None = None,
        load_states: bool = False,
        original_lattice_update: bool = True,
        t_emb_dim: int = 512,
        condition_keys: list[str] | None = None,
        condition_dim: int | None = None,
        max_atomic_number: int = 100,
        atom_emb_dim: int = 6272,
        edge_emb_dim: int = 128,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        regress_forces: bool = True,
        otf_graph: bool = False,
        max_neighbors: int = 20,
        max_radius: float = 12.0,
        max_num_elements: int | None = None,
        num_layers: int = 20,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 64,
        num_heads: int = 8,
        attn_alpha_channels: int = 64,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 128,
        norm_type: str = "layer_norm_sh",
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        grid_resolution: int | None = 18,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = True,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.1,
        proj_drop: float = 0.0,
        weight_init: str = "uniform",
        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: float | None = None,
        avg_degree: float | None = None,
        use_energy_lin_ref: bool | None = False,
        load_energy_lin_ref: bool | None = False,
        activation_checkpoint: bool = False,
    ):
        if not original_lattice_update:
            raise ValueError(
                "EquiformerV2Wrapper only supports original_lattice_update=True."
            )

        if condition_keys is None:
            condition_keys = []
        if condition_dim is None:
            condition_dim = t_emb_dim

        if lmax_list is None:
            lmax_list = [6]
        if mmax_list is None:
            mmax_list = [3]
        if max_num_elements is None:
            max_num_elements = max_atomic_number + 1

        super().__init__(
            use_pbc=use_pbc,
            use_pbc_single=use_pbc_single,
            regress_forces=regress_forces,
            otf_graph=otf_graph,
            max_neighbors=max_neighbors,
            max_radius=max_radius,
            max_num_elements=max_num_elements,
            num_layers=num_layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            norm_type=norm_type,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            grid_resolution=grid_resolution,
            num_sphere_samples=num_sphere_samples,
            edge_channels=edge_channels,
            use_atom_edge_embedding=use_atom_edge_embedding,
            share_atom_edge_embedding=share_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,
            attn_activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            ffn_activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,
            weight_init=weight_init,
            enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
            avg_num_nodes=avg_num_nodes,
            avg_degree=avg_degree,
            use_energy_lin_ref=use_energy_lin_ref,
            load_energy_lin_ref=load_energy_lin_ref,
            activation_checkpoint=activation_checkpoint,
        )

        expected_atom_emb_dim = self._get_atom_feature_dim()
        if atom_emb_dim != expected_atom_emb_dim:
            raise ValueError(
                f"atom_emb_dim={atom_emb_dim} does not match "
                f"Equiformer output size {expected_atom_emb_dim}."
            )
        if edge_emb_dim != self.edge_channels:
            raise ValueError(
                f"edge_emb_dim={edge_emb_dim} does not match "
                f"Equiformer edge size {self.edge_channels}."
            )

        self.time_embedding = nn.Linear(t_emb_dim, self.sphere_channels)
        self.angle_edge_emb = nn.Sequential(
            nn.Linear(self.edge_channels + 3, self.edge_channels),
            nn.SiLU(),
            nn.Linear(self.edge_channels, self.edge_channels),
        )

        self.initial_edge_mlp = RadialFunction(list(self.edge_channels_list))
        edge_channels_list = list(self.edge_channels_list)
        edge_channels_list[0] = (
            edge_channels_list[0] + self.edge_channels + 2 * self.sphere_channels
        )
        self.edge_embedding_layers = nn.ModuleList(
            [RadialFunction(list(edge_channels_list)) for _ in range(self.num_layers)]
        )
        self.lattice_out_blocks = nn.ModuleList(
            [
                EquiformerLatticeUpdateBlockFrac(
                    edge_emb_dim=self.edge_channels,
                    num_rbfs=int(self.distance_expansion.num_output),
                )
                for _ in range(self.num_layers + 1)
            ]
        )
        self.energy_head = EqV2ScalarHead(self, output_name="energy", reduce="sum")
        self.force_head = EqV2VectorHead(self, output_name="forces")

        self._set_adapter_embedding(condition_keys, condition_dim)

        if checkpoint_path is not None and load_states:
            self._load_checkpoint(checkpoint_path)

    def _get_atom_feature_dim(self) -> int:
        # Atom-type logits should be predicted from invariant scalar channels only.
        return self.sphere_channels

    def _set_adapter_embedding(
        self,
        condition_keys: list[str],
        condition_dim: int,
    ):
        cond_projectors = {}
        cond_adapters_per_layer = {}
        for cond_name in condition_keys:
            cond_projectors[cond_name] = nn.Linear(
                condition_dim,
                self.sphere_channels,
            )
            cond_adapters_per_layer[cond_name] = nn.ModuleList(
                [ControlNetAdapter(self.sphere_channels) for _ in range(self.num_layers)]
            )

        self.cond_projectors = nn.ModuleDict(cond_projectors)
        self.cond_adapters_per_layer = nn.ModuleDict(cond_adapters_per_layer)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._remap_legacy_checkpoint_keys(rename_keys(checkpoint["state_dict"]))
        model_state = self.state_dict()

        adapted_state_dict = {}
        skipped_keys = []
        for key, value in state_dict.items():
            if key not in model_state:
                continue

            target = model_state[key]
            if value.shape == target.shape:
                adapted_state_dict[key] = value
                continue

            if (
                value.ndim == target.ndim
                and value.shape[1:] == target.shape[1:]
                and value.shape[0] <= target.shape[0]
            ):
                padded = target.clone()
                padded[: value.shape[0]] = value
                adapted_state_dict[key] = padded
                continue

            skipped_keys.append(key)

        keys_info = self.load_state_dict(adapted_state_dict, strict=False)

        missing_keys = [
            key
            for key in keys_info.missing_keys
            if not key.startswith("time_embedding")
            and not key.startswith("cond_projectors")
            and not key.startswith("cond_adapters_per_layer")
            and not key.startswith("initial_edge_mlp")
            and not key.startswith("edge_embedding_layers")
            and not key.startswith("angle_edge_emb")
            and not key.startswith("lattice_out_blocks")
        ]

        if keys_info.unexpected_keys:
            warnings.warn(
                f"Unexpected keys when loading checkpoint {checkpoint_path}: "
                f"{keys_info.unexpected_keys}",
                stacklevel=2,
            )
        if missing_keys:
            warnings.warn(
                f"Missing keys when loading checkpoint {checkpoint_path}: {missing_keys}",
                stacklevel=2,
            )
        if skipped_keys:
            warnings.warn(
                f"Skipped incompatible checkpoint tensors from {checkpoint_path}: "
                f"{skipped_keys}",
                stacklevel=2,
            )

    def _remap_legacy_checkpoint_keys(self, state_dict):
        remapped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("energy_block."):
                remapped_state_dict[f"energy_head.{key}"] = value
            elif key.startswith("force_block."):
                remapped_state_dict[f"force_head.{key}"] = value
            else:
                remapped_state_dict[key] = value
        return remapped_state_dict

    def _build_node_embeddings(
        self,
        atomic_numbers: torch.Tensor,
        batch: torch.Tensor,
        t_emb: torch.Tensor,
        edge_distance_features: torch.Tensor,
        graph,
    ) -> SO3_Embedding:
        x = SO3_Embedding(
            len(atomic_numbers),
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        edge_degree = self.edge_degree_embedding(
            graph.atomic_numbers_full,
            edge_distance_features,
            graph.edge_index,
            len(atomic_numbers),
            graph.node_offset,
        )
        x.embedding = x.embedding + edge_degree.embedding
        x.embedding[:, 0, :] = x.embedding[:, 0, :] + self.time_embedding(t_emb[batch])
        return x

    def _prepare_conditioning(self, batch, added_cond=None, cond_mask=None):
        if added_cond is None:
            return None, None

        added_cond_per_atom = {
            name: self.cond_projectors[name](cond[batch])
            for name, cond in added_cond.items()
        }

        if cond_mask is not None:
            cond_mask_per_atom = {
                name: cond[batch].float() for name, cond in cond_mask.items()
            }
        else:
            cond_mask_per_atom = {
                name: torch.ones_like(cond[:, :1]) for name, cond in added_cond_per_atom.items()
            }

        return added_cond_per_atom, cond_mask_per_atom

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
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        atomic_numbers = data.atomic_numbers.long()

        data.build_graph(
            cutoff=self.max_radius,
            max_neighbors=self.max_neighbors,
        )

        if atomic_numbers.max().item() >= self.max_num_elements:
            raise ValueError(
                f"Atomic number {atomic_numbers.max().item()} exceeds "
                f"max_num_elements={self.max_num_elements}."
            )

        graph = self.generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        if graph.edge_index.numel() == 0:
            raise RuntimeError(
                "EquiformerV2Wrapper: graph has no edges inside the cutoff. "
                f"max_radius={self.max_radius}, max_neighbors={self.max_neighbors}."
            )

        data_batch = data.batch
        if gp_utils.initialized():
            (
                atomic_numbers,
                data_batch,
                node_offset,
                edge_index,
                edge_distance,
                edge_distance_vec,
            ) = self._init_gp_partitions(
                graph.atomic_numbers_full,
                graph.batch_full,
                graph.edge_index,
                graph.edge_distance,
                graph.edge_distance_vec,
            )
            graph.node_offset = node_offset
            graph.edge_index = edge_index
            graph.edge_distance = edge_distance
            graph.edge_distance_vec = edge_distance_vec

        graph_edge_distances = graph.edge_distance
        distance_basis = self.distance_expansion(graph.edge_distance)
        edge_distance_features = distance_basis
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = graph.atomic_numbers_full[graph.edge_index[0]]
            target_element = graph.atomic_numbers_full[graph.edge_index[1]]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance_features = torch.cat(
                (distance_basis, source_embedding, target_embedding), dim=1
            )
        graph.edge_distance = edge_distance_features

        edge_rot_mat = self._init_edge_rot_mat(
            data, graph.edge_index, graph.edge_distance_vec
        )
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        x = self._build_node_embeddings(
            atomic_numbers=atomic_numbers,
            batch=data_batch,
            t_emb=t_emb,
            edge_distance_features=edge_distance_features,
            graph=graph,
        )

        added_cond_per_atom, cond_mask_per_atom = self._prepare_conditioning(
            batch=data_batch,
            added_cond=added_cond,
            cond_mask=cond_mask,
        )

        distance_vec = graph.edge_distance_vec
        batch_edge = data_batch[graph.edge_index[0]]
        edge_unit_vec = distance_vec / graph_edge_distances[:, None].clamp(min=1e-8)
        cosines = torch.cosine_similarity(
            edge_unit_vec[:, None],
            data.cell[batch_edge],
            dim=-1,
        )

        x_edge = self.initial_edge_mlp(edge_distance_features)
        edge_states = [x_edge] if output_edges_hidden_states else None
        node_states = [x.embedding[:, 0, :]] if output_nodes_hidden_states else None

        lattice_update = self.lattice_out_blocks[0](
            edge_emb=self.angle_edge_emb(torch.cat([x_edge, cosines], dim=-1)),
            edge_index=graph.edge_index,
            distance_vec=distance_vec,
            batch=data_batch,
            rbf=distance_basis,
            normalize_score=True,
        )

        for i in range(self.num_layers):
            if added_cond_per_atom is not None:
                scalar_state = x.embedding[:, 0, :]
                h_adapt = torch.zeros_like(scalar_state)

                for cond_name, cond in added_cond_per_atom.items():
                    h_adapt_cond = self.cond_adapters_per_layer[cond_name][i](
                        scalar_state,
                        cond,
                    )
                    h_adapt += cond_mask_per_atom[cond_name] * h_adapt_cond

                x.embedding[:, 0, :] = scalar_state + h_adapt

            x = self.blocks[i](
                x,
                graph.atomic_numbers_full,
                edge_distance_features,
                graph.edge_index,
                batch=data_batch,
                node_offset=graph.node_offset,
            )

            x_0_embedding = x.embedding[:, 0, :]
            source_x_0_embedding = x_0_embedding[graph.edge_index[0]]
            target_x_0_embedding = x_0_embedding[graph.edge_index[1]]

            edge_layer_input = torch.cat(
                [
                    edge_distance_features,
                    x_edge,
                    source_x_0_embedding,
                    target_x_0_embedding,
                ],
                dim=1,
            )
            x_edge = x_edge + self.edge_embedding_layers[i](edge_layer_input)

            lattice_update = lattice_update + self.lattice_out_blocks[i + 1](
                edge_emb=self.angle_edge_emb(torch.cat([x_edge, cosines], dim=-1)),
                edge_index=graph.edge_index,
                distance_vec=distance_vec,
                batch=data_batch,
                rbf=distance_basis,
                normalize_score=True,
            )

            if output_nodes_hidden_states:
                node_states.append(x.embedding[:, 0, :])
            if output_edges_hidden_states:
                edge_states.append(x_edge)

        x.embedding = self.norm(x.embedding)

        emb = {"node_embedding": x, "graph": graph}
        outputs = self.energy_head(data, emb)
        if self.regress_forces:
            outputs.update(self.force_head(data, emb))

        outputs["nodes_last_hidden_state"] = x.embedding[:, 0, :]
        outputs["graph_edge_index"] = graph.edge_index
        outputs["graph_edge_distance_vec"] = graph.edge_distance_vec
        outputs["graph_edge_distances"] = graph_edge_distances
        outputs["lattice_update"] = lattice_update

        if output_nodes_hidden_states:
            outputs["nodes_hidden_states"] = tuple(node_states)
        if output_edges_hidden_states:
            outputs["edges_hidden_states"] = tuple(edge_states)

        return outputs

    def save_pretrained(
        self,
        save_directory,
        is_main_process: bool = True,
        save_function=None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size="10GB",
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
