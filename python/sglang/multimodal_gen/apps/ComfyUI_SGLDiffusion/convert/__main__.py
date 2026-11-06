# SPDX-License-Identifier: Apache-2.0
"""python -m sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.convert IN.json [-o OUT.json] [--report R.md]

Converts a ComfyUI workflow (UI or API JSON) to run its H3 / Qwen / Z-Image / FLUX.1
model on SGLD. Writes IN_sgld.json next to the input by default and prints the report.
"""

import argparse
import json
import os
import sys

from .workflow import convert_api, convert_workflow, dumps, is_api_prompt


def _model_lookup(dirs):
    def find(name):
        if not isinstance(name, str):
            return None
        for d in dirs:
            for root, _, files in os.walk(d, followlinks=True):
                if os.path.basename(name.replace("\\", "/")) in files:
                    return os.path.join(root, os.path.basename(name.replace("\\", "/")))
        return None

    return find


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workflow")
    ap.add_argument("-o", "--output")
    ap.add_argument("--report", help="also write the report (markdown) here")
    ap.add_argument(
        "--models",
        action="append",
        default=[],
        help="diffusion_models directory, to read model headers (FastH3 detection)",
    )
    a = ap.parse_args(argv)
    data = json.load(open(a.workflow, encoding="utf-8"))
    convert = convert_api if is_api_prompt(data) else convert_workflow
    out, report = convert(data, _model_lookup(a.models) if a.models else None)
    path = a.output or os.path.splitext(a.workflow)[0] + "_sgld.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps(out))
    text = report.markdown()
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.write(text + f"\nwrote {path}\n")


if __name__ == "__main__":
    main()
