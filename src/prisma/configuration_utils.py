import json
from collections.abc import Mapping
from typing import Any

from diffusers.configuration_utils import (
    ConfigMixin as DiffusersConfigMixin,
    FrozenDict,
)
from omegaconf import OmegaConf

from prisma import __version__


def _to_plain_config(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(
            value,
            resolve=True,
            throw_on_missing=True,
            enum_to_str=True,
        )

    if isinstance(value, Mapping):
        return {key: _to_plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_config(item) for item in value]

    return value


class ConfigMixin(DiffusersConfigMixin):
    def register_to_config(self, **kwargs) -> None:
        super().register_to_config(**_to_plain_config(kwargs))

    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        if hasattr(self, "_internal_dict"):
            self._internal_dict = FrozenDict(_to_plain_config(self._internal_dict))

        json_s = super().to_json_string()

        config_dict = json.loads(json_s)

        config_dict["_prisma_version"] = __version__

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
