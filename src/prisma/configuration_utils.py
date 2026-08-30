import json

from diffusers.configuration_utils import ConfigMixin as DiffusersConfigMixin

from prisma import __version__


class ConfigMixin(DiffusersConfigMixin):
    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        json_s = super().to_json_string()

        config_dict = json.loads(json_s)

        config_dict["_prisma_version"] = __version__

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
