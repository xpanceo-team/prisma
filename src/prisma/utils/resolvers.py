from omegaconf import ListConfig, OmegaConf
import torch

from prisma import __version__


def _resolve_device_multiplier(devices) -> int:
    if isinstance(devices, int):
        if devices == -1:
            return max(1, torch.cuda.device_count())
        if devices <= 0:
            return 1
        return devices

    if isinstance(devices, str):
        stripped = devices.strip()
        if stripped.isdigit():
            return max(1, int(stripped))
        if stripped == "auto":
            return max(1, torch.cuda.device_count())
        return 1

    if isinstance(devices, (list, tuple, set, ListConfig)):
        return max(1, len(devices))

    return 1


def get_effective_batch_size(batch_size, grad_accum, devices):
    return batch_size * _resolve_device_multiplier(devices) * grad_accum


def register_resolvers():
    resolvers = {
        "multiply": lambda x, y: x * y,
        "effective_batch_size": get_effective_batch_size,
        "version": lambda: __version__,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)
