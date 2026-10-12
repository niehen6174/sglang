"""Add the derivable rope.inv_freq buffer that the FastH3 8-Step V2 Diffusers export drops.

Mirrors the transformer step of kevin-mi/FastH3-8step-V2-overlay without running its code.
Usage: materialize-fasth3-v2-transformer.py MODEL_DIR  (MODEL_DIR/transformer holds symlinked shards)
"""
import json, os, sys

import torch
from safetensors.torch import save_file

transformer_dir = os.path.join(sys.argv[1], "transformer")
with open(os.path.join(transformer_dir, "config.json")) as f:
    rope_freq_dim = int(json.load(f)["rope_freq_dim"])
exponents = torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)
save_file({"rope.inv_freq": 1.0 / (10000.0**exponents)}, os.path.join(transformer_dir, "sglang_rope_inv_freq.safetensors"))
index_path = os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)
index["weight_map"]["rope.inv_freq"] = "sglang_rope_inv_freq.safetensors"
os.remove(index_path)
with open(index_path, "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)
print("rope.inv_freq", rope_freq_dim)
