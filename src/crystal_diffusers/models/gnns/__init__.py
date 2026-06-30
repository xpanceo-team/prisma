from crystal_diffusers.models.gnns.gemnet import GemNetTWrapper
from crystal_diffusers.models.gnns.equiformer_v2 import EquiformerV2Wrapper

__all__ = ["GemNetTWrapper", "EquiformerV2Wrapper"]

try:
    from crystal_diffusers.models.gnns.pet import PETMADWrapper, PETWrapper
except ImportError:
    # PET wrapper has optional dependencies (metatrain/metatomic).
    pass
else:
    __all__.extend(["PETWrapper", "PETMADWrapper"])
