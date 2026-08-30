from prisma.models.gnns.gemnet import GemNetTWrapper
from prisma.models.gnns.equiformer_v2 import EquiformerV2Wrapper

__all__ = ["GemNetTWrapper", "EquiformerV2Wrapper"]

try:
    from prisma.models.gnns.pet import PETMADWrapper, PETWrapper
except ImportError:
    # PET wrapper has optional dependencies (metatrain/metatomic).
    pass
else:
    __all__.extend(["PETWrapper", "PETMADWrapper"])
