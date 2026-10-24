# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the opt-in example; no model runtime is imported."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXAMPLE = (
    Path(__file__).resolve().parents[5] / "examples/usage/h3_cpu_frame_delivery/run.py"
)
SPEC = importlib.util.spec_from_file_location("h3_cpu_frame_delivery", EXAMPLE)
EXAMPLE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE_MODULE)


class TestH3CPUFrameDelivery(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference image.png"
        self.reference.write_bytes(b"unused by configuration-only tests")

    def build(self, baseline=False, reference=None):
        return EXAMPLE_MODULE.build_config(
            self.root / "model",
            "test prompt",
            self.root / "out",
            reference_image=reference,
            baseline=baseline,
        )

    def test_h3_changes_only_the_explicit_frame_delivery_flag(self):
        baseline = self.build(True, self.reference)
        optimized = self.build(False, self.reference)
        self.assertEqual(baseline["engine_kwargs"], optimized["engine_kwargs"])
        self.assertTrue(optimized["sampling_params_kwargs"]["return_frames"])
        baseline["sampling_params_kwargs"]["return_frames"] = True
        self.assertEqual(baseline, optimized)

    def test_h3_keeps_required_native_files_and_reference_workload(self):
        config = self.build(reference=self.reference)
        params = config["sampling_params_kwargs"]
        self.assertTrue(params["save_output"])
        self.assertFalse(params["return_file_paths_only"])
        self.assertTrue(params["output_path"])
        self.assertEqual(params["reference_resize_mode"], "max")
        self.assertEqual(params["quality"], "exact")
        self.assertEqual(
            params["conditions"][0]["uri"], self.reference.resolve().as_uri()
        )
        self.assertNotIn("return_raw_frames", params)

    def test_missing_h3_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference-image"):
            self.build()

    def test_dry_run_imports_no_runtime_and_creates_no_outputs(self):
        sentinel = self.root / "sglang"
        sentinel.mkdir()
        (sentinel / "__init__.py").write_text(
            "raise AssertionError('runtime import is forbidden')"
        )
        env = dict(os.environ, PYTHONPATH=str(self.root))
        result = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE),
                "--model-path",
                str(self.root / "missing-model"),
                "--prompt",
                "test",
                "--output-dir",
                str(self.root / "out"),
                "--reference-image",
                str(self.reference),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["engine_kwargs"]["model_id"], "MiniMax-H3"
        )
        self.assertFalse((self.root / "out").exists())

    def test_run_rejects_missing_weights_before_importing_runtime(self):
        result = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE),
                "--model-path",
                str(self.root / "missing-model"),
                "--prompt",
                "test",
                "--run",
                "--reference-image",
                str(self.reference),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("existing local model", result.stderr)


if __name__ == "__main__":
    unittest.main()
