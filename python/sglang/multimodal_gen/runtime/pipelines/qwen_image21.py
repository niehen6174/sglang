# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
    QwenImage21PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.qwenimage21 import QwenImage21SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21DenoisingStage,
    QwenImage21EncodingStage,
    QwenImage21InputValidationStage,
    prepare_qwen21_mu,
)


class QwenImage21Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "QwenImage21Pipeline"
    # Config for a ComfyUI single-file DiT, which has no model_index.json.
    pipeline_config_cls = QwenImage21PipelineConfig
    sampling_params_cls = QwenImage21SamplingParams
    _required_config_modules = [
        "processor",
        "text_encoder",
        "transformer",
        "vae",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args):
        self.add_stage(QwenImage21InputValidationStage())
        self.add_stage_factory(
            RoleType.ENCODER,
            lambda: QwenImage21EncodingStage(
                self.get_module("text_encoder"),
                self.get_module("processor"),
                self.get_module("vae"),
                self.get_module("scheduler"),
            ),
            "conditioning_stage",
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[prepare_qwen21_mu]
        )
        self.add_stage_factory(
            RoleType.DENOISER,
            lambda: QwenImage21DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "denoising_stage",
        )
        self.add_standard_decoding_stage()

    def create_comfyui_stages(self, server_args):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_cfg_split import (
            ComfyUICFGSplitStage,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21_comfyui import (
            QwenImage21ComfyUIStepStage,
        )

        stage = QwenImage21ComfyUIStepStage(
            transformer=self.get_module("transformer"),
            scheduler=self.get_module("scheduler"),
        )
        if server_args.enable_cfg_parallel:
            if server_args.sp_degree > 1 or server_args.tp_size > 1:
                raise ValueError(
                    "Qwen-Image 2.1 CFG split in --comfyui-mode runs one cond per "
                    "GPU; combining it with sp_degree / tp_size is not supported"
                )
            # ComfyUI owns CFG: each CFG rank runs one of the step's conds.
            stage = ComfyUICFGSplitStage(stage)
        self.add_stage(stage)


EntryClass = QwenImage21Pipeline
