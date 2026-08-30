from diffusers.pipelines.pipeline_utils import (
    DiffusionPipeline as DiffusersDiffusionPipeline,
)

from prisma.pipelines.pipeline_loading_utils import LOADABLE_CLASSES


assert "prisma" in LOADABLE_CLASSES


class DiffusionPipeline(DiffusersDiffusionPipeline):
    pass
