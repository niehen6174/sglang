# SPDX-License-Identifier: Apache-2.0
"""Explicit MOSS SFT snapshot-offload example."""

import argparse
import json
from pathlib import Path


def build_config(model_path, prompt, output_dir, baseline=False):
    engine = dict(
        model_path=str(model_path),
        num_gpus=1,
        performance_mode="manual",
        warmup_mode="off",
        disable_conditioning_cache=True,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        dit_cpu_offload=True,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pin_cpu_memory=True,
        dit_precision="bf16",
    )
    sampling = dict(
        prompt=prompt,
        seed=150000,
        generator_device="cpu",
        num_inference_steps=50,
        save_output=True,
        output_path=str(output_dir),
    )
    engine.update(
        model_id="MOSS-Image-SFT",
        attention_backend="torch_sdpa",
        image_encoder_cpu_offload=False,
        disable_autocast=True,
        vae_precision="fp32",
        vae_tiling=False,
    )
    if not baseline:
        engine["component_residency"] = {
            "dit": "snapshot-offload",
            "text_encoder": "snapshot-offload",
        }
    sampling.update(
        negative_prompt="",
        width=1024,
        height=1024,
        guidance_scale=3.0,
        flow_shift=5.0,
        max_sequence_length=4096,
        uncond_skip_layers=[22],
        slg_layers=[],
        slg_scale=0.0,
        slg_start=0.01,
        slg_stop=1.0,
    )
    return dict(engine_kwargs=engine, sampling_params_kwargs=sampling)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--baseline", action="store_true", help="Undo only the measured profile change"
    )
    parser.add_argument(
        "--run", action="store_true", help="Generate once using an existing local model"
    )
    args = parser.parse_args(argv)
    try:
        config = build_config(
            args.model_path, args.prompt, args.output_dir, args.baseline
        )
    except ValueError as error:
        parser.error(str(error))
    if not args.run:
        print(json.dumps(config, indent=2))
        return
    if not args.model_path.is_dir():
        parser.error("--run requires an existing local model directory")
    from sglang.multimodal_gen import DiffGenerator

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = DiffGenerator.from_pretrained(**config["engine_kwargs"])
    try:
        result = generator.generate(
            sampling_params_kwargs=config["sampling_params_kwargs"]
        )
        results = result if isinstance(result, list) else [result]
        paths = [getattr(item, "output_file_path", None) for item in results]
        if not paths or any((not path or not Path(path).is_file() for path in paths)):
            raise RuntimeError(
                "The native generator did not produce the requested files"
            )
        print(json.dumps(dict(output_files=paths), indent=2))
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
