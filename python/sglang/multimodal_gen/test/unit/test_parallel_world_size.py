"""Every GPU must belong to the dp*tp*sp*cfg layout.

A surplus rank got no process group and deadlocked worker startup in
torch.distributed.new_group instead of failing.
"""

import unittest
from unittest.mock import patch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, auto_tune


class TestParallelWorldSize(unittest.TestCase):
    def _args(self, **kwargs):
        # Memory auto-tuning probes every requested GPU; CI runs this on one.
        with (
            patch.object(
                PipelineConfig, "from_kwargs", return_value=QwenImagePipelineConfig()
            ),
            patch.object(
                auto_tune.ServerArgsAutoTuner,
                "_get_min_available_device_memory_gb",
                return_value=80.0,
            ),
        ):
            return ServerArgs.from_dict(
                {"model_path": "test/model", "cfg_parallel_degree": 1, **kwargs}
            )

    def test_surplus_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must equal dp_size"):
            self._args(num_gpus=2, tp_size=1, sp_degree=1)

    def test_layout_larger_than_world_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must equal dp_size"):
            self._args(num_gpus=2, tp_size=2, sp_degree=2)

    def test_exact_layouts_are_accepted(self):
        for kwargs in (
            {"tp_size": 2},
            {"sp_degree": 2, "ulysses_degree": 2},
            {"dp_size": 2},
            {},
        ):
            with self.subTest(kwargs=kwargs):
                args = self._args(num_gpus=2, **kwargs)
                self.assertEqual(
                    args.dp_size * args.tp_size * args.sp_degree, args.num_gpus
                )
