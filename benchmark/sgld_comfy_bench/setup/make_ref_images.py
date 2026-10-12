"""Provenance for inputs/ref_01.jpg .. ref_08.jpg (the H3 Ref2VA references).

They were generated once with stock ComfyUI + Qwen-Image 2.1 (bf16, 768x768, 20 steps, cfg 1,
seed 7000 + i) and committed; machines use the committed files and never re-run this.
Usage (only to regenerate): GPUS=<one gpu> python setup/make_ref_images.py
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
import comfyserver as comfy  # noqa: E402
import graphs  # noqa: E402
import models  # noqa: E402

SUBJECTS = [
    "an orange tabby cat sitting, full body",
    "a golden retriever dog standing, full body",
    "a young woman in a red wool coat, half body portrait",
    "an old man wearing a brown fedora hat, half body portrait",
    "a blue vintage car from the 1960s, side view",
    "an acoustic guitar standing upright",
    "a small wooden cabin house with a green door",
    "a friendly silver humanoid robot, full body",
]


def main():
    out_dir = pathlib.Path(__file__).resolve().parents[1] / "inputs"
    suite = pathlib.Path(os.environ["WORKSPACE"]) / "results" / "make_refs"
    srv, mon = comfy.Server("gpu0", suite), comfy.make_monitor("gpu0")
    srv.start("", {})
    try:
        for i, subject in enumerate(SUBJECTS):
            g = graphs.qwen("stock", models.qwen_files("h200"), None, width=768, height=768, steps=20, cfg=1.0,
                            seed=7000 + i, prefix=f"refs/ref_{i + 1:02d}")
            for n in g.values():
                if n["class_type"] == "TextEncodeQwenImage21":
                    n["inputs"]["prompt"] = f"Studio photo of {subject}, plain light grey background, soft light, sharp focus."
            res = comfy.run_prompt(srv, g, f"ref_{i + 1:02d}", mon, 900, keep_media=True)
            png = next((suite / "runs" / f"ref_{i + 1:02d}").glob("*.png"))
            from PIL import Image
            Image.open(png).convert("RGB").save(out_dir / f"ref_{i + 1:02d}.jpg", quality=93)
            print(i + 1, res.get("status"), res.get("e2e_s"), flush=True)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
