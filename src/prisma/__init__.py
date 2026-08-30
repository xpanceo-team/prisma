__version__ = "0.1.0"


import warnings


# Suppress known pymatgen warnings for elements without Pauling electronegativity
warnings.filterwarnings(
    "ignore",
    message=r"No Pauling electronegativity for .*",
    category=UserWarning,
    module=r"pymatgen\.core\.(composition|periodic_table)",
)

# import logging first to configure it
import prisma.utils.logging

from prisma.configuration_utils import ConfigMixin
from prisma.pipelines import DiffusionPipeline, MatterGenPipeline
from prisma.models import (
    ModelMixin,
    MatterGenModel,
    ConditionEncoder,
    GemNetTWrapper,
    EquiformerV2Wrapper,
)

try:
    from prisma.models import PETMADWrapper, PETWrapper
except ImportError:
    # PET wrapper has optional dependencies.
    pass

from prisma.schedulers import (
    SchedulerMixin,
    D3PMScheduler,
    VarianceExplodingScheduler,
    VariancePreservingScheduler,
)
from prisma.utils.resolvers import register_resolvers

register_resolvers()
