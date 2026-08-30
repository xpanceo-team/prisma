from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES


LOADABLE_CLASSES["prisma"] = {
    "ModelMixin": ["save_pretrained", "from_pretrained"],
    "SchedulerMixin": ["save_pretrained", "from_pretrained"],
    "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
}
