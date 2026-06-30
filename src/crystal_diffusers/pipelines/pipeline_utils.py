from diffusers.pipelines.pipeline_utils import (
    DiffusionPipeline as DiffusersDiffusionPipeline,
)

from crystal_diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES


assert "crystal_diffusers" in LOADABLE_CLASSES


class DiffusionPipeline(DiffusersDiffusionPipeline):
    pass
