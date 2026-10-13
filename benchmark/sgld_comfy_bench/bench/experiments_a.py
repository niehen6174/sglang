"""Track A experiment registry: SGLD integrated mode vs stock ComfyUI (EXPERIMENTS.md sections 4-5).

Each experiment resolves, for one card, to a ComfyUI graph, the ComfyUI command-line flags, extra
environment for the ComfyUI process (inherited by the SGLD worker), the number of GPUs and the
warmup step count. "BEST" stands for the card's best lossless SGLD attention backend, written by
the P0 sweep to SUITE/best_attention.json.
"""
import json
import os
import pathlib

import graphs
import models

SAGE2, SAGE3 = "sage_attn", "sage_attn_3"
STOCK_SAGE = "--use-sage-attention"
# ComfyUI cannot see the worker's memory; on 24/32 GB cards it must drop TE/VAE after use.
SGLD_SMALL_FLAGS = "--disable-smart-memory"
DEFAULT_BEST = {"4090": "fa", "5090": "torch_cudnn_sdpa", "h200": "fa", "b200": "torch_cudnn_sdpa"}
P0_BACKENDS = ["fa", "torch_sdpa", "torch_cudnn_sdpa"]
KITCHEN_ENV = {"SGLANG_DIFFUSION_CONVROT_INT8_BACKEND": "comfy_kitchen"}

H3_768 = dict(width=1344, height=768, length=124)
H3_480 = dict(width=864, height=480, length=124)
H3_SWEEP = dict(width=864, height=480, length=39, steps=8)


def u2():
    return {"num_gpus": 2, "tp_size": 1, "sp_degree": 2, "ulysses_degree": 2, "ring_degree": 1}


def tp2():
    return {"num_gpus": 2, "tp_size": 2, "sp_degree": 1, "ulysses_degree": 1, "ring_degree": 1}


def best_attention(card, suite):
    f = pathlib.Path(suite) / "best_attention.json"
    if f.exists():
        return json.loads(f.read_text())["best"]
    return DEFAULT_BEST[card]


def _exp(eid, model, engine, *, cards=models.CARDS, n_gpu=1, kind="perf", **spec):
    return dict(id=eid, track="A", model=model, engine=engine, cards=tuple(cards), n_gpu=n_gpu, kind=kind, spec=spec)


def _h3_list():
    big, small, bw = models.BIG, models.SMALL, ("5090", "b200")
    E = []
    for be in P0_BACKENDS + [SAGE2, SAGE3]:
        cards = bw if be == SAGE3 else models.CARDS
        E.append(_exp(f"P0-H3-{be}", "h3", "sgld", cards=cards, kind="p0", canvas=H3_SWEEP, attn=be))
    E += [
        _exp("A-H3-01", "h3", "stock", canvas=H3_768),
        _exp("A-H3-02", "h3", "stock", canvas=H3_768, flags=STOCK_SAGE),
        _exp("A-H3-03", "h3", "stock", cards=("h200", "5090"), canvas=H3_768, easycache=True),
        _exp("A-H3-04", "h3", "stock", cards=big, canvas=H3_768, unet="dit_int8"),
        _exp("A-H3-10", "h3", "sgld", canvas=H3_768, attn=""),
        _exp("A-H3-11", "h3", "sgld", canvas=H3_768, attn="BEST"),
        _exp("A-H3-20", "h3", "sgld", canvas=H3_768, attn=SAGE2),
        _exp("A-H3-21", "h3", "sgld", cards=bw, canvas=H3_768, attn=SAGE3),
        _exp("A-H3-22", "h3", "sgld", canvas=H3_768, attn="BEST", cache_dit=True),
        _exp("A-H3-23", "h3", "sgld", cards=big, canvas=H3_768, attn="BEST", unet="dit_int8"),
        _exp("A-H3-24", "h3", "sgld", cards=big, canvas=H3_768, attn="BEST", unet="dit_int8", env=KITCHEN_ENV),
        _exp("A-H3-25", "h3", "sgld", canvas=H3_768, attn="BEST", variant="fast_h3"),
        _exp("A-H3-26", "h3", "sgld", canvas=H3_768, variant="vdn_h3"),
        _exp("A-H3-27", "h3", "sgld", canvas=H3_768, attn=SAGE2, unet="dit_int8", cache_dit=True),
        _exp("A-H3-28", "h3", "sgld", canvas=H3_768, attn=SAGE2, variant="fast_h3", int8=True),
        _exp("A-H3-30", "h3", "sgld", n_gpu=2, canvas=H3_768, attn="BEST", par="u2"),
        _exp("A-H3-31", "h3", "sgld", n_gpu=2, canvas=H3_768, attn="BEST", par="tp2"),
        _exp("A-H3-32", "h3", "sgld", n_gpu=2, canvas=H3_768, attn=SAGE2, unet="dit_int8", cache_dit=True, par="u2"),
        _exp("A-H3-33", "h3", "sgld", n_gpu=2, canvas=H3_768, attn="BEST", variant="fast_h3", par="u2"),
        _exp("A-H3-34", "h3", "sgld", n_gpu=2, canvas=H3_768, variant="vdn_h3", par="u2"),  # VDN supports Ulysses, not TP
        _exp("A-H3-40", "h3", "stock", canvas=H3_768, refs=4),
        _exp("A-H3-41", "h3", "sgld", canvas=H3_768, attn="BEST", refs=4),
        _exp("A-H3-42", "h3", "sgld", canvas=H3_768, attn=SAGE2, cache_dit=True, refs=4),
        _exp("A-H3-43", "h3", "sgld", n_gpu=2, canvas=H3_768, attn="BEST", refs=4, par="u2|tp2"),
    ]
    # 480p supplement on 4090/5090 for the starred configs.
    for base in ("A-H3-01", "A-H3-11", "A-H3-25", "A-H3-26", "A-H3-27", "A-H3-30"):
        e = next(x for x in E if x["id"] == base)
        E.append(dict(e, id=base + "-480", cards=tuple(c for c in e["cards"] if c in small),
                      spec=dict(e["spec"], canvas=H3_480)))
    return E


def _ltx_list():
    big, bw = models.BIG, ("5090", "b200")
    return [
        _exp("A-LTX-01", "ltx", "stock"),
        _exp("A-LTX-02", "ltx", "stock", flags=STOCK_SAGE),
        _exp("A-LTX-10", "ltx", "sgld", attn=""),
        _exp("A-LTX-11", "ltx", "sgld", attn="BEST"),
        _exp("A-LTX-20", "ltx", "sgld", attn=SAGE2),
        _exp("A-LTX-21", "ltx", "sgld", cards=bw, attn=SAGE3),
        _exp("A-LTX-23", "ltx", "sgld", cards=big, attn="BEST", unet="dit_bf16"),
        _exp("A-LTX-30", "ltx", "sgld", n_gpu=2, attn="BEST", par="u2"),
        _exp("A-LTX-31", "ltx", "sgld", n_gpu=2, attn=SAGE2, par="u2"),
    ]


def _qwen_list():
    bw = ("5090", "b200")
    E = []
    for cfg in (1.0, 4.0):
        s = f"-cfg{int(cfg)}"
        E += [
            _exp("A-QW-01" + s, "qwen", "stock", cfg=cfg),
            _exp("A-QW-02" + s, "qwen", "stock", cfg=cfg, flags=STOCK_SAGE),
            _exp("A-QW-10" + s, "qwen", "sgld", cfg=cfg, attn=""),
            _exp("A-QW-11" + s, "qwen", "sgld", cfg=cfg, attn="BEST"),
            _exp("A-QW-20" + s, "qwen", "sgld", cfg=cfg, attn=SAGE2),
            _exp("A-QW-21" + s, "qwen", "sgld", cfg=cfg, cards=bw, attn=SAGE3),
            _exp("A-QW-23" + s, "qwen", "sgld", cfg=cfg, attn="BEST", unet="dit_int8"),
            _exp("A-QW-30" + s, "qwen", "sgld", cfg=cfg, n_gpu=2, attn="BEST", par="u2"),
            _exp("A-QW-31" + s, "qwen", "sgld", cfg=cfg, n_gpu=2, attn="BEST", par="tp2"),
        ]
    return E


EXPERIMENTS = _h3_list() + _ltx_list() + _qwen_list()


def offload_default(card, model):
    """Whether the SGLD DiT starts with layerwise offload on this card (OOM retries with it on)."""
    if card == "4090":
        return model in ("h3", "ltx")
    return False


def _sgld_options(e, card, suite):
    s = e["spec"]
    opts = {"model_type": "auto-detect", "dit_layerwise_offload": offload_default(card, e["model"])}
    attn = s.get("attn", "")
    opts["attention_backend"] = best_attention(card, suite) if attn == "BEST" else attn
    if s.get("cache_dit"):
        opts["enable_cache_dit"] = True
    variant = s.get("variant")
    if variant:
        opts["model_type"] = variant
        opts["runtime_model_path"] = models.FASTH3_DIR() if variant == "fast_h3" else models.VDN_DIR()
        if variant == "vdn_h3":
            opts["attention_backend"] = "hybrid_window_attn_h3"
        # The distilled DiTs exist only as 66 GB BF16; small cards need online INT8 plus offload.
        if card in models.SMALL:
            opts.update(quantization="convrot_int8", dit_layerwise_offload=True)
        elif s.get("int8"):
            opts["quantization"] = "convrot_int8"
        elif variant == "vdn_h3":
            # VDN's config maps "none" to BF16; left empty it would pick MXFP8 on Blackwell.
            opts["quantization"] = "none"
    par = s.get("par")
    if par:
        choice = par.split("|")
        mode = choice[0] if card in models.BIG or len(choice) == 1 else choice[1]
        opts.update(u2() if mode == "u2" else tp2())
    return opts


def resolve(e, card, suite):
    """-> dict(graph, flags, env, warmup_steps, model, video_seconds) for this card."""
    s, model, engine = e["spec"], e["model"], e["engine"]
    sgld = _sgld_options(e, card, suite) if engine == "sgld" else None
    flags = s.get("flags", "")
    if engine == "sgld" and card in models.SMALL:
        flags = (flags + " " + SGLD_SMALL_FLAGS).strip()
    env = dict(s.get("env", {}))
    prefix = f"bench/{e['id']}"
    if model == "h3":
        files = models.h3_files(card)
        if s.get("unet"):
            files["unet_override"] = files[s["unet"]]
        if s.get("variant") and engine == "sgld":
            files["unet_override"] = files["dit_int8"]  # loader needs an H3 file; the worker uses runtime_model_path
        c = dict(s["canvas"])
        steps = c.pop("steps", 20)
        graph = graphs.h3(engine, files, sgld, task="r2v" if s.get("refs") else "t2v", steps=steps,
                          distilled=s.get("variant"), refs=s.get("refs", 0), prefix=prefix, **c)
        if s.get("easycache"):
            graph = _add_easycache(graph)
        warm = None if s.get("variant") else 3
        seconds = c["length"] / 24.0
    elif model == "ltx":
        files = models.ltx_files(card)
        graph = graphs.ltx(engine, files, sgld, unet=files.get(s["unet"]) if s.get("unet") else None, prefix=prefix)
        warm, seconds = None, 121 / 24.0
    else:
        files = models.qwen_files(card)
        graph = graphs.qwen(engine, files, sgld, cfg=s["cfg"], unet=files.get(s["unet"]) if s.get("unet") else None,
                            prefix=prefix)
        warm, seconds = 3, None
    return dict(graph=graph, flags=flags, env=env, warmup_steps=warm, model=model, video_seconds=seconds,
                sgld_options=sgld)


def _add_easycache(graph):
    """Stock ComfyUI's EasyCache node between the model and the guider (default threshold)."""
    model_ref = next(k for k, n in graph.items() if n["class_type"] == "H3BenchmarkModel")
    key = str(max(int(k) for k in graph) + 1)
    for n in graph.values():
        if n["class_type"] in ("BasicGuider", "CFGGuider") and n["inputs"].get("model") == [model_ref, 0]:
            n["inputs"]["model"] = [key, 0]
    graph[key] = {"class_type": "EasyCache", "inputs": {"model": [model_ref, 0], "reuse_threshold": 0.2,
                                                         "start_percent": 0.15, "end_percent": 0.95, "verbose": False}}
    return graph


def for_card(card, n_gpu=None, only=None, kinds=("p0", "perf")):
    out = []
    for e in EXPERIMENTS:
        if card not in e["cards"] or e["kind"] not in kinds:
            continue
        if n_gpu is not None and e["n_gpu"] != n_gpu:
            continue
        if only and not any(e["id"] == o or e["id"].startswith(o.rstrip("*")) and o.endswith("*") for o in only):
            continue
        out.append(e)
    return out


if __name__ == "__main__":
    import sys

    card = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CARD", "h200")
    for e in for_card(card):
        print(f"{e['id']:16s} {e['model']:5s} {e['engine']:6s} gpus={e['n_gpu']} {e['spec']}")
