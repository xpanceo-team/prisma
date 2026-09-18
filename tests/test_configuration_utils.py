import json

from diffusers.configuration_utils import FrozenDict
from omegaconf import OmegaConf

from prisma.configuration_utils import ConfigMixin


class _Configurable(ConfigMixin):
    config_name = "config.json"


def test_config_mixin_serializes_omegaconf_as_plain_json_values():
    configurable = _Configurable()
    configurable._internal_dict = FrozenDict(
        {
            "condition_keys": OmegaConf.create(["bandgap_hse", "dn", "shg"]),
            "condition": OmegaConf.create({"dn": {"scale": True}}),
        }
    )

    config = json.loads(configurable.to_json_string())

    assert config["condition_keys"] == ["bandgap_hse", "dn", "shg"]
    assert config["condition"] == {"dn": {"scale": True}}
