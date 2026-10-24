# SPDX-License-Identifier: Apache-2.0
"""Explicit, measured diffusion configurations; prints JSON unless --run is set."""

import argparse
import json
from pathlib import Path

PROFILES = ("h3-ref2va-cpu-frames", "moss-sft-snapshot")


def build_config(
    profile, model_path, prompt, output_dir, reference_image=None, baseline=False
):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    if profile == "moss-sft-snapshot" and reference_image is not None:
        raise ValueError(
            "The MOSS SFT profile is text-to-image; omit --reference-image"
        )

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
    if profile == "h3-ref2va-cpu-frames":
        if reference_image is None or not Path(reference_image).is_file():
            raise ValueError("H3 Ref2VA requires an existing --reference-image")
        engine.update(
            model_id="MiniMax-H3",
            backend="sglang",
            pipeline_class_name="MiniMaxH3Pipeline",
            model_variant="ref2va",
            attention_backend="fa",
            vae_tiling=True,
        )
        sampling.update(
            task="ref2va",
            conditions=[
                dict(
                    type="image",
                    role="reference",
                    uri=Path(reference_image).resolve().as_uri(),
                )
            ],
            target=dict(aspect_ratio="16:9", duration_seconds=5.0, short_edge=768),
            width=1344,
            height=768,
            flow_shift=12.0,
            audio_flow_shift=3.0,
            quality="exact",
            reference_resize_mode="max",
            enable_cache_dit=False,
            # This profile explicitly requests CPU frames as well as native files.
            # File-only consumers should retain the native paths-only default.
            return_file_paths_only=False,
            return_frames=not baseline,
        )
    else:
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
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--reference-image", type=Path)
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
            args.profile,
            args.model_path,
            args.prompt,
            args.output_dir,
            args.reference_image,
            args.baseline,
        )
    except ValueError as error:
        parser.error(str(error))
    if not args.run:
        print(json.dumps(config, indent=2))
        return
    if not args.model_path.is_dir():
        parser.error("--run requires an existing local model directory")

    # Configuration inspection and unit tests do not import the model runtime.
    from sglang.multimodal_gen import DiffGenerator

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = DiffGenerator.from_pretrained(**config["engine_kwargs"])
    try:
        result = generator.generate(
            sampling_params_kwargs=config["sampling_params_kwargs"]
        )
        results = result if isinstance(result, list) else [result]
        paths = [getattr(item, "output_file_path", None) for item in results]
        if not paths or any(not path or not Path(path).is_file() for path in paths):
            raise RuntimeError(
                "The native generator did not produce the requested files"
            )
        print(json.dumps(dict(output_files=paths), indent=2))
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
