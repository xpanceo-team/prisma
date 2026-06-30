__version__ = "0.1.0"


import warnings


# Ignore FutureWarning messages from any module starting with 'fairchem.'
warnings.filterwarnings("ignore", category=FutureWarning, module=r"fairchem\.")

# Suppress known pymatgen warnings for elements without Pauling electronegativity
warnings.filterwarnings(
    "ignore",
    message=r"No Pauling electronegativity for .*",
    category=UserWarning,
    module=r"pymatgen\.core\.(composition|periodic_table)",
)

# import logging first to configure it
import crystal_diffusers.utils.logging

from crystal_diffusers.configuration_utils import ConfigMixin
from crystal_diffusers.pipelines import DiffusionPipeline, MatterGenPipeline
from crystal_diffusers.models import (
    ModelMixin,
    MatterGenModel,
    ConditionEncoder,
    GemNetTWrapper,
    EquiformerV2Wrapper,
)

try:
    from crystal_diffusers.models import PETMADWrapper, PETWrapper
except ImportError:
    # PET wrapper has optional dependencies.
    pass

from crystal_diffusers.schedulers import (
    SchedulerMixin,
    D3PMScheduler,
    VarianceExplodingScheduler,
    VariancePreservingScheduler,
)
from crystal_diffusers.utils.resolvers import register_resolvers

register_resolvers()
