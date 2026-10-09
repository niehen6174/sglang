"""ComfyUI integrated mode must reject data parallelism up front.

Its per-step requests reached a different replica than the one holding the
run's cached conditioning, then crashed on CUDA IPC inputs from another device.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


@unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two visible GPUs")
class TestComfyUIModeDataParallel(unittest.TestCase):
    def _args(self, **kwargs):
        with patch.object(
            PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
        ):
            return ServerArgs.from_dict({"model_path": "test/model", **kwargs})

    def test_dp_rejected_in_comfyui_mode(self):
        with self.assertRaisesRegex(ValueError, "ComfyUI integrated mode"):
            self._args(num_gpus=2, dp_size=2, comfyui_mode=True)

    def test_dp_still_allowed_outside_comfyui_mode(self):
        self.assertEqual(self._args(num_gpus=2, dp_size=2).dp_size, 2)
