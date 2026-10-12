"""Track A functional matrix (EXPERIMENTS.md 5.4): does each case run, on stock ComfyUI and on SGLD.

One run per case and engine, short settings (H3 at 864x480, 39 frames, 8 steps), no timing.
Outcome per cell: PASS, FAIL (with the error), or UNSUPPORTED (the engine has no way to express it).
Cases that also appear in the perf matrix (FastH3/VDN, multi-GPU) are read from the track A
results when present instead of re-run.

Usage: functional.py --suite DIR [--only F-01a,F-09*]
"""
import argparse
import json
import os
import pathlib

import comfyserver as comfy
import experiments_a as A
import graphs
import models

KIT = pathlib.Path(__file__).resolve().parents[1]
SHORT = dict(width=864, height=480, length=39, steps=8)
LORA_TURBO8 = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def _h3(engine, card, sgld, **kw):
    files = models.h3_files(card)
    files["dit"] = files["dit_int8"]
    args = {**SHORT, **kw}
    return graphs.h3(engine, files, sgld, prefix="video/bench/F", **args)


def _patch_h3(graph, **inputs):
    for n in graph.values():
        if n["class_type"] in ("MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"):
            n["inputs"].update(inputs)
    return graph


def _with_lora(graph, engine):
    """Turbo 8-step LoRA between the loader and H3BenchmarkModel."""
    loader = next(k for k, n in graph.items() if n["class_type"] in ("SGLDUNETLoader", "UNETLoader"))
    key = str(max(int(k) for k in graph) + 1)
    if engine == "sgld":
        graph[key] = {"class_type": "SGLDLoraLoader", "inputs": {"model": [loader, 0], "lora_name": LORA_TURBO8,
                                                                  "strength_model": 1.0, "nickname": "turbo8", "target": "all"}}
    else:
        graph[key] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [loader, 0], "lora_name": LORA_TURBO8,
                                                                       "strength_model": 1.0}}
    for k, n in graph.items():
        if k != key and n["inputs"].get("model") == [loader, 0]:
            n["inputs"]["model"] = [key, 0]
    return graph


def _ltx_i2v(engine, card, sgld):
    """LTX-2.5 two-stage with the first frame injected by LTXVImgToVideoInplace (stage 1)."""
    g = graphs.ltx(engine, models.ltx_files(card), sgld, prefix="video/bench/F")
    vae = next(k for k, n in g.items() if n["class_type"] == "VAELoader" and "audio" not in n["inputs"]["vae_name"])
    vlat = next(k for k, n in g.items() if n["class_type"] == "EmptyLTXVLatentVideo")
    img, i2v = str(max(int(k) for k in g) + 1), str(max(int(k) for k in g) + 2)
    g[img] = {"class_type": "LoadImage", "inputs": {"image": "cmp_cat_first.png"}}
    g[i2v] = {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": [vae, 0], "image": [img, 0], "latent": [vlat, 0],
                                                                "strength": 1.0, "bypass": False}}
    for n in g.values():
        if n["class_type"] == "LTXVConcatAVLatent" and n["inputs"].get("video_latent") == [vlat, 0]:
            n["inputs"]["video_latent"] = [i2v, 0]
    return g


def _qwen_edit2(engine, card, sgld):
    g = graphs.qwen(engine, models.qwen_files(card), sgld, edit=True, prefix="image/bench/F")
    img = str(max(int(k) for k in g) + 1)
    g[img] = {"class_type": "LoadImage", "inputs": {"image": "ref_02.jpg"}}
    for n in g.values():
        if n["class_type"] == "TextEncodeQwenImage21":
            n["inputs"]["images.image_2"] = [img, 0]
            n["inputs"]["prompt"] = "Put the cat from image 1 next to the dog from image 2 on a sofa, watercolor style."
    return g


def cases(card):
    """(case_id, description, builder(engine) -> graph or None if unsupported, sgld options)."""
    h3o = {"model_type": "auto-detect", "attention_backend": "", "dit_layerwise_offload": A.offload_default(card, "h3")}
    qo = {"model_type": "auto-detect", "attention_backend": ""}
    lo = {"model_type": "auto-detect", "attention_backend": "", "dit_layerwise_offload": A.offload_default(card, "ltx")}
    return [
        ("F-01a", "H3 T2V", lambda e: _h3(e, card, h3o)),
        ("F-01b", "H3 FL2VA first+last frame", lambda e: _fl2v(e, card, h3o)),
        ("F-01c", "H3 Ref2VA, 1 reference", lambda e: _h3(e, card, h3o, task="r2v", refs=1)),
        ("F-02", "H3 + turbo 8-step LoRA", lambda e: _with_lora(_h3(e, card, h3o), e)),
        ("F-03", "H3 odd size 1000x560", lambda e: _h3(e, card, h3o, width=1000, height=560)),
        ("F-04a", "H3 FastH3 (distilled)", lambda e: None if e == "stock" else _h3(
            e, card, {**h3o, **A._sgld_options(next(x for x in A.EXPERIMENTS if x["id"] == "A-H3-25"), card, "")},
            distilled="fast_h3")),
        ("F-04b", "H3 VDN (distilled)", lambda e: None if e == "stock" else _h3(
            e, card, {**h3o, **A._sgld_options(next(x for x in A.EXPERIMENTS if x["id"] == "A-H3-26"), card, "")},
            distilled="vdn_h3")),
        ("F-05", "H3 2-GPU Ulysses", lambda e: None if e == "stock" else _h3(e, card, {**h3o, **A.u2()})),
        ("F-06a", "LTX-2.5 T2V two-stage", lambda e: graphs.ltx(e, models.ltx_files(card), lo, prefix="video/bench/F")),
        ("F-06b", "LTX-2.5 I2V two-stage", lambda e: _ltx_i2v(e, card, lo)),
        ("F-07", "LTX-2.3 FP8 all-in-one + distilled LoRA", lambda e: json.loads(
            (KIT / "workflows" / f"ltx23_fp8_lora_{e}.json").read_text())),
        ("F-08", "LTX-2.5 2-GPU Ulysses", lambda e: None if e == "stock" else graphs.ltx(
            e, models.ltx_files(card), {**lo, **A.u2()}, prefix="video/bench/F")),
        ("F-09a", "Qwen T2I INT8", lambda e: graphs.qwen(e, models.qwen_files(card), qo, unet=models.QWEN["dit_int8"],
                                                         prefix="image/bench/F")),
        ("F-09b", "Qwen edit, 1 reference", lambda e: graphs.qwen(e, models.qwen_files(card), qo, edit=True,
                                                                 prefix="image/bench/F")),
        ("F-09c", "Qwen edit, 2 references", lambda e: _qwen_edit2(e, card, qo)),
        ("F-09d", "Qwen odd size 1040x1024", lambda e: graphs.qwen(e, models.qwen_files(card), qo, width=1040,
                                                                  prefix="image/bench/F")),
        ("F-10", "Qwen 2-GPU Ulysses", lambda e: None if e == "stock" else graphs.qwen(
            e, models.qwen_files(card), {**qo, "num_gpus": 2, "sp_degree": 2, "ulysses_degree": 2},
            prefix="image/bench/F")),
    ]


def _fl2v(engine, card, sgld):
    g = _h3(engine, card, sgld)
    first, last = str(max(int(k) for k in g) + 1), str(max(int(k) for k in g) + 2)
    g[first] = {"class_type": "LoadImage", "inputs": {"image": "cmp_cat_first.png"}}
    g[last] = {"class_type": "LoadImage", "inputs": {"image": "cmp_cat_last.png"}}
    return _patch_h3(g, first_frame=[first, 0], last_frame=[last, 0])


def run(card, suite, only, timeout):
    out = []
    for engine in ("stock", "sgld"):
        for cid, desc, build in cases(card):
            if only and not any(cid == o or (o.endswith("*") and cid.startswith(o[:-1])) for o in only):
                continue
            graph = build(engine)
            row = {"case": cid, "desc": desc, "engine": engine, "card": card}
            if graph is None:
                row.update(outcome="UNSUPPORTED")
            else:
                n_gpu = 2 if cid in ("F-05", "F-08", "F-10") else 1
                lane = "all" if n_gpu == 2 else "gpu0"
                flags = A.SGLD_SMALL_FLAGS if engine == "sgld" and card in models.SMALL else ""
                srv, mon = comfy.Server(lane, suite), comfy.make_monitor(lane)
                try:
                    srv.start(flags, {}, monitor=mon)
                    res = comfy.run_prompt(srv, graph, f"{cid}__{engine}_00", mon, timeout, keep_media=True)
                    row.update(outcome="PASS" if res.get("status") == "success" else "FAIL", e2e_s=res.get("e2e_s"),
                               error=(res.get("error") or "")[:800], media=res.get("media"))
                except Exception as ex:  # noqa: BLE001
                    row.update(outcome="FAIL", error=f"{type(ex).__name__}: {ex}"[:800])
                finally:
                    srv.stop()
            print(f"{cid:6s} {engine:5s} {row['outcome']:11s} {desc}  {row.get('error', '')[:150]}", flush=True)
            with open(suite / "functional.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            out.append(row)
    # Model switching in one ComfyUI process: H3 -> Qwen -> H3, SGLD only.
    if not only or any(o.startswith("F-11") for o in only):
        srv, mon = comfy.Server("gpu0", suite), comfy.make_monitor("gpu0")
        h3o = {"model_type": "auto-detect", "attention_backend": "", "dit_layerwise_offload": A.offload_default(card, "h3")}
        seq = [("h3", _h3("sgld", card, h3o)), ("qwen", graphs.qwen("sgld", models.qwen_files(card),
                                                                    {"model_type": "auto-detect"}, prefix="image/bench/F")),
               ("h3", _h3("sgld", card, h3o))]
        row = {"case": "F-11", "desc": "switch H3 -> Qwen -> H3 in one ComfyUI", "engine": "sgld", "card": card, "steps": []}
        try:
            srv.start(A.SGLD_SMALL_FLAGS if card in models.SMALL else "", {}, monitor=mon)
            for i, (name, g) in enumerate(seq):
                res = comfy.run_prompt(srv, g, f"F-11__{i}_{name}", mon, timeout)
                row["steps"].append({"model": name, "status": res.get("status"), "peak_vram_gib": res.get("peak_vram_gib"),
                                     "error": (res.get("error") or "")[:300]})
            row["outcome"] = "PASS" if all(s["status"] == "success" for s in row["steps"]) else "FAIL"
        except Exception as ex:  # noqa: BLE001
            row.update(outcome="FAIL", error=f"{type(ex).__name__}: {ex}"[:800])
        finally:
            srv.stop()
        print(f"F-11   sgld  {row['outcome']}", flush=True)
        with open(suite / "functional.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--only", type=lambda s: s.split(","))
    ap.add_argument("--force", action="store_true", help="run on a card outside the plan (h200, 5090)")
    a = ap.parse_args()
    if os.environ["CARD"] not in models.FUNCTIONAL_CARDS and not a.force:
        print(f"functional matrix runs on {models.FUNCTIONAL_CARDS} only (use --force)")
        return
    suite = pathlib.Path(a.suite).resolve()
    (suite / "runs").mkdir(parents=True, exist_ok=True)
    run(os.environ["CARD"], suite, a.only, float(os.environ.get("TIMEOUT_S", "5400")))
    print("FUNCTIONAL_DONE", flush=True)


if __name__ == "__main__":
    main()
