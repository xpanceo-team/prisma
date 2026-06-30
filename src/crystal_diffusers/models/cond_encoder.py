import torch
from torch import nn

from diffusers.configuration_utils import register_to_config

from crystal_diffusers.models.embeddings import (
    NoiseLevelEncoding,
    SpaceGroupEncoding,
    ChemicalSystemEncoding,
    VectorEncoding,
    CategoricalEncoding,
)
from crystal_diffusers.models.modeling_utils import ModelMixin
from crystal_diffusers.models.scaler import StandardScaler
from crystal_diffusers.configuration_utils import ConfigMixin


class ConditionEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        t_emb_dim: int = 512,
        condition_dim: int = 512,
        condition: dict[str, any] | None = None,
        dropout: float = 0.1,
        scale_t_emb: float = 1.0,
        bias_t_emb: float = 0.0,
    ):
        super().__init__()

        if condition is None:
            condition = {}

        self.register_to_config(condition=condition)

        self.condition_keys = list(condition.keys())

        self.timestep_encoding = NoiseLevelEncoding(t_emb_dim)

        self._set_condition_embeddings()
        self._set_scalers()

        self.register_buffer("scale_t", torch.tensor(scale_t_emb))
        self.register_buffer("bias_t", torch.tensor(bias_t_emb))

    def forward(
        self, condition: dict[str, any], t: torch.LongTensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        t_emb = self.timestep_encoding(t * self.scale_t + self.bias_t)

        added_cond = {}
        added_mask_cond = {}
        for cond_name in self.condition_keys:
            if cond_name not in condition:
                continue

            encoding = self.condition_encdoings[cond_name]
            condition_value = condition[cond_name]
            cond_mask = self._get_cond_mask(len(condition_value), t.device)

            if torch.is_tensor(condition_value):
                needs_scaling = self.config.condition[cond_name].get("scale", False)
                if needs_scaling:
                    condition_value = self.scalers[cond_name](condition_value)

                nan_mask = condition_value.isnan()
                nan_mask = nan_mask if nan_mask.ndim == 1 else nan_mask.any(
                    axis=1 if nan_mask.ndim == 2 else -1
                )
                cond_mask = cond_mask & (~nan_mask.unsqueeze(1))

                condition_value = condition_value.clone()
                condition_value[nan_mask] = 0.0
                cond_encoded = encoding(condition_value)
            else:
                cond_encoded = encoding(condition_value)

            added_mask_cond[cond_name] = cond_mask

            cond_encoded = cond_mask.float() * cond_encoded

            cond_type = self.config.condition[cond_name]["condition_type"]
            if cond_type == "adapter":
                added_cond[cond_name] = cond_encoded
            elif cond_type == "time_emb":
                t_emb += cond_encoded
            else:
                raise ValueError(f"Unknown condition type: {cond_type}")

        return t_emb, added_cond, added_mask_cond

    def _get_cond_mask(
        self, batch_size: int, device: torch.device | str
    ) -> torch.Tensor:
        if not self.training or self.config.dropout == 0:
            cond_mask = torch.ones(batch_size, device=device)
        else:
            cond_mask = torch.rand(batch_size, device=device) > self.config.dropout

        return cond_mask.bool().unsqueeze(1)

    def _set_condition_embeddings(self):
        encodings = {}
        for cond_name in self.condition_keys:
            cond_info = self.config.condition[cond_name]

            encoding_type = cond_info["encoding_type"]

            cond_type = cond_info["condition_type"]

            if cond_type == "adapter":
                cond_dim = self.config.condition_dim
            elif cond_type == "time_emb":
                cond_dim = self.config.t_emb_dim
            else:
                raise ValueError(f"Unknown condition type: {cond_type}")

            if encoding_type == "vector":
                encoding_module = VectorEncoding(
                    input_dim=cond_info["input_dim"],
                    hidden_dim=cond_dim,
                )
            elif encoding_type == "categorical":
                encoding_module = CategoricalEncoding(
                    num_categories=cond_info["num_categories"],
                    hidden_dim=cond_dim,
                )
            elif encoding_type == "chemical_system":
                encoding_module = ChemicalSystemEncoding(
                    input_dim=cond_info["input_dim"],
                    hidden_dim=cond_dim,
                )
            elif encoding_type == "space_group":
                encoding_module = SpaceGroupEncoding(cond_dim)
            elif encoding_type == "sinusoidal":
                encoding_module = NoiseLevelEncoding(cond_dim)
            else:
                raise ValueError(f"Unknown encoding type: {encoding_type}")

            encodings[cond_name] = encoding_module

        self.condition_encdoings = nn.ModuleDict(encodings)

    def _set_scalers(self):
        scalers = {}
        for cond_name in self.condition_keys:
            cond_info = self.config.condition[cond_name]

            if not cond_info.get("scale"):
                continue

            mean = cond_info.get("scale_mean")
            std = cond_info.get("scale_std")

            if not mean or not std:
                raise ValueError(
                    f"Condition `{cond_name}` requires scaling "
                    f"(scale={cond_info['scale']} specified), but mean or std weren't "
                    f"specified in config."
                )

            scaler = StandardScaler(mean=mean, std=std)

            scalers[cond_name] = scaler

        self.scalers = nn.ModuleDict(scalers)
