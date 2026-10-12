"""Independent re-implementation of the VDN-H3 transformer prefuse.

Mirrors kevin-mi/VDN-H3-overlay@7de1827 _prefuse_transformer + _write_rope_inv_freq
(read and audited, not executed). Output is checked against the handoff's
sglang_vdn_prefused.json / index / config afterwards.
"""

import hashlib
import json
import os
import pathlib
import sys
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = pathlib.Path(os.environ["WORKSPACE"])
SRC = pathlib.Path(os.environ["HF_HUB_CACHE"]) / (
    "models--OpenVDN--vdn-minimax-h3/snapshots/"
    "50744d80bde8c1adbc5f37089f125e0e436bf7b0"
)
OUT = ROOT / "models/VDN-H3-8nfe/transformer"
CKPT = SRC / "stage-dmd-step-250"
BASE = SRC / "h3-base/transformer"
ORDER = ("default", "turbo")
EXPECTED_PAIRS = {"default": 208, "turbo": 363}
EXPECTED_LINEAR_KEYS = 800
LINEAR_FILE = "sglang_vdn_linear_branch.safetensors"
ROPE_FILE = "sglang_rope_inv_freq.safetensors"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def lora_target(key):
    kind = "A" if ".lora_A." in key else "B" if ".lora_B." in key else None
    if kind is None:
        raise ValueError(key)
    base = key.split(f".lora_{kind}.")[0].replace(".attn.orig.", ".attn.")
    return base + ".weight", kind


def load_adapter(name):
    d = CKPT / "adapters" / name
    cfg_path = d / "adapter_spec.json"
    if not cfg_path.exists():
        cfg_path = d / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())["config"]
    pairs = defaultdict(dict)
    with safe_open(d / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            target, kind = lora_target(key)
            pairs[target][kind] = f.get_tensor(key)
    for target, pair in pairs.items():
        assert set(pair) == {"A", "B"}, target
        module = target[: -len(".weight")]
        rank = int(cfg.get("rank_pattern", {}).get(module, cfg["rank"]))
        alpha = float(cfg.get("alpha_pattern", {}).get(module, cfg["alpha"]))
        assert pair["A"].shape[0] == rank and pair["B"].shape[1] == rank, target
        pair["scale"] = alpha / rank
    record = {
        "name": name,
        "path": str(d.relative_to(SRC)),
        "pairs": len(pairs),
        "rank": int(cfg["rank"]),
        "alpha": float(cfg["alpha"]),
        "rank_pattern_modules": len(cfg.get("rank_pattern", {})),
        "scales": sorted({p["scale"] for p in pairs.values()}),
        "adapter_model_sha256": sha256(d / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(cfg_path),
    }
    assert len(pairs) == EXPECTED_PAIRS[name], (name, len(pairs))
    return pairs, record


def main():
    torch.set_num_threads(16)
    OUT.mkdir(parents=True, exist_ok=True)
    base_config = json.loads((BASE / "config.json").read_text())
    base_index = json.loads(
        (BASE / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    weight_map = dict(base_index["weight_map"])
    deltas = defaultdict(list)
    adapters = []
    for name in ORDER:
        pairs, record = load_adapter(name)
        for target, pair in pairs.items():
            assert target in weight_map, target
            deltas[target].append((pair["A"], pair["B"], pair["scale"]))
        adapters.append(record)
    merged_pairs = merged_tensors = 0
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(BASE / shard, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for name in f.keys():
                t = f.get_tensor(name)
                if name in deltas:
                    m = t.to(torch.float32)
                    for a, b, scale in deltas[name]:
                        delta = b.to(torch.float32) @ a.to(torch.float32)
                        assert delta.shape == m.shape, name
                        m.add_(delta, alpha=scale)
                        merged_pairs += 1
                    t = m.to(t.dtype)
                    merged_tensors += 1
                tensors[name] = t.contiguous()
        target = OUT / shard
        if target.is_symlink() or target.exists():
            target.unlink()
        save_file(tensors, str(target), metadata=metadata or {"format": "pt"})
        print("wrote", shard, flush=True)
        del tensors
    assert merged_pairs == sum(EXPECTED_PAIRS.values()), merged_pairs

    branch_src = CKPT / "linear_branch/model.safetensors"
    branch_cfg = json.loads((CKPT / "linear_branch/config.json").read_text())
    assert branch_cfg.get("type") == "hybrid_attention" and branch_cfg.get("version") == 2
    link = OUT / LINEAR_FILE
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.realpath(branch_src))
    with safe_open(branch_src, framework="pt", device="cpu") as f:
        branch_keys = list(f.keys())
    assert len(branch_keys) == EXPECTED_LINEAR_KEYS, len(branch_keys)
    for key in branch_keys:
        assert key not in weight_map, key
        weight_map[key] = LINEAR_FILE

    config = dict(base_config)
    config["hybrid_attention"] = branch_cfg["config"]
    (OUT / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    meta = base_index.get("metadata") or {}
    index = {
        "metadata": {
            **meta,
            "total_size": int(meta.get("total_size", 0)) + branch_src.stat().st_size,
        },
        "weight_map": weight_map,
    }

    rope_dim = int(base_config["rope_freq_dim"])
    exps = torch.arange(0, 2 * rope_dim, 2, dtype=torch.float32) / (2 * rope_dim)
    save_file({"rope.inv_freq": 1.0 / (10000.0**exps)}, str(OUT / ROPE_FILE))
    index["weight_map"]["rope.inv_freq"] = ROPE_FILE
    (OUT / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True)
    )

    record = {
        "source_model_id": "OpenVDN/vdn-minimax-h3",
        "vdn_checkpoint": "stage-dmd-step-250",
        "checkpoint_metadata": json.loads((CKPT / "metadata.json").read_text()),
        "merge": {
            "rule": "W += (alpha / rank) * B @ A, fp32 accumulate, one rounding to the stored dtype",
            "order": list(ORDER),
            "pairs_merged": merged_pairs,
            "tensors_touched": merged_tensors,
        },
        "adapters": adapters,
        "linear_branch": {
            "file": LINEAR_FILE,
            "tensors": len(branch_keys),
            "sha256": sha256(branch_src),
            "config": branch_cfg,
        },
    }
    (OUT / "sglang_vdn_prefused.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)
    )
    print(json.dumps(record["merge"]), "DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
