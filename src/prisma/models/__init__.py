from prisma.models.modeling_utils import ModelMixin
from prisma.models.cond_encoder import ConditionEncoder
from prisma.models.mattergen import MatterGenModel
from prisma.models.gnns import GemNetTWrapper, EquiformerV2Wrapper

__all__ = [
    "ModelMixin",
    "ConditionEncoder",
    "MatterGenModel",
    "GemNetTWrapper",
    "EquiformerV2Wrapper",
]

try:
    from prisma.models.gnns import PETMADWrapper, PETWrapper
except ImportError:
    # PET wrapper has optional dependencies (metatrain/metatomic).
    pass
else:
    __all__.extend(["PETWrapper", "PETMADWrapper"])
