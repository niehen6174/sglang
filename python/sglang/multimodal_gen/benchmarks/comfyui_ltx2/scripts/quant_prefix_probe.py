"""Which prefix does each LTX-2 DiT linear pass to the quant config? (base vs branch)

Builds a tiny LTX-2.5 DiT with (a) a recording quant config and (b) online
Fp8Config(ignored_layers=...), and prints per-linear prefix / chosen method.
usage: PYTHONPATH=<tree>/python python quant_prefix_probe.py OUT.json
"""
import json, os, sys
import torch
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29613", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
from sglang.multimodal_gen.runtime.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel
maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
import importlib.util, sglang.multimodal_gen as _mg
_spec = importlib.util.spec_from_file_location("unit_conftest", os.path.join(os.path.dirname(_mg.__file__), "test/unit/conftest.py"))
_conf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_conf)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
set_global_server_args(_conf._make_unit_server_args())
from sglang.multimodal_gen.configs.models.dits.ltx_2_5 import LTX25Config
from sglang.multimodal_gen.runtime.layers.linear import LinearBase, UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import LTX2VideoTransformer3DModel

TINY = dict(num_attention_heads=2, attention_head_dim=16, cross_attention_dim=32,
            audio_num_attention_heads=2, audio_attention_head_dim=8, audio_cross_attention_dim=16,
            caption_channels=32, num_layers=1, in_channels=16, out_channels=16,
            connector_attention_head_dim=16, connector_num_attention_heads=2,
            audio_connector_attention_head_dim=8, audio_connector_num_attention_heads=2)


def tiny_config():
    cfg = LTX25Config()
    for k, v in TINY.items():
        if hasattr(cfg.arch_config, k):
            setattr(cfg.arch_config, k, v)
    cfg.arch_config.__post_init__()
    return cfg


class Recording(QuantizationConfig):
    def __init__(self):
        super().__init__(); self.seen = []
    @classmethod
    def get_name(cls): return "recording"
    @classmethod
    def get_supported_act_dtypes(cls): return [torch.bfloat16, torch.float32]
    @classmethod
    def get_min_capability(cls): return 0
    @staticmethod
    def get_config_filenames(): return []
    @classmethod
    def from_config(cls, config): return cls()
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            self.seen.append(prefix); return UnquantizedLinearMethod()
        return None
    def get_scaled_act_names(self): return []


rec = Recording()
with torch.device("meta"):
    LTX2VideoTransformer3DModel(config=tiny_config(), hf_config={}, quant_config=rec)
ignored = ["transformer_blocks.0.attn1", "proj_out", "transformer_blocks.0.ff.proj_out"]
fp8 = Fp8Config(ignored_layers=ignored)
methods = {}
with torch.device("meta"):
    m = LTX2VideoTransformer3DModel(config=tiny_config(), hf_config={}, quant_config=fp8)
for name, mod in m.named_modules():
    if isinstance(mod, LinearBase):
        methods[name] = type(mod.quant_method).__name__
out = {"prefixes": rec.seen, "fp8_ignored": ignored, "fp8_methods": methods}
json.dump(out, open(sys.argv[1], "w"), indent=1)
empty = sum(1 for p in rec.seen if not p)
unq = sorted(n for n, k in methods.items() if k == "UnquantizedLinearMethod")
print(f"linears={len(rec.seen)} empty_prefix={empty} fp8_unquantized={unq}")

# GGUF: an LTX GGUF usable in ComfyUI (city96 ComfyUI-GGUF) carries ComfyUI
# state-dict names; GGUFConfig looks up f"{prefix}.weight" verbatim. Compare the
# prefixes of a real-size LTX-2.3 DiT with the real ComfyUI checkpoint names.
ckpt = "/scratch/data/sgld_comfy/ComfyUI-ltx25/models/diffusion_models/ltx-2.3-22b-dev-fp8.safetensors"
try:
    from safetensors import safe_open
    from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.ltx_2 import (
        apply_comfyui_ltx_config, read_comfyui_ltx_checkpoint_info)
except ImportError:
    sys.exit(0)  # base tree: no ComfyUI LTX spec
with safe_open(ckpt, "pt") as f:
    names = {k.removeprefix("model.diffusion_model.") for k in f.keys()}
cfg = LTX25Config()
apply_comfyui_ltx_config(cfg.arch_config, read_comfyui_ltx_checkpoint_info(ckpt))
cfg.arch_config.embeddings_connectors_in_dit = False  # native layout: connectors are a separate component
rec_full = Recording()
with torch.device("meta"):
    LTX2VideoTransformer3DModel(config=cfg, hf_config={}, quant_config=rec_full)
missing = [p for p in rec_full.seen if f"{p}.weight" not in names]
kinds = sorted({p.split(".", 2)[-1] if p.startswith("transformer_blocks.") else p for p in missing})
out["gguf_full_linears"] = len(rec_full.seen)
out["gguf_missing"] = missing
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"full-size linears={len(rec_full.seen)} not found under ComfyUI names={len(missing)} kinds={kinds}")
