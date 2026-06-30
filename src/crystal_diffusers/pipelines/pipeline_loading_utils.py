from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES


LOADABLE_CLASSES["crystal_diffusers"] = {
    "ModelMixin": ["save_pretrained", "from_pretrained"],
    "SchedulerMixin": ["save_pretrained", "from_pretrained"],
    "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
}
