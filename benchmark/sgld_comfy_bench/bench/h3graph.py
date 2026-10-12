"""Build ComfyUI API-format MiniMax-H3 graphs for the test campaign.

One config dict describes a run; `build(cfg)` returns the API graph. engine="official"
uses UNETLoader/LoraLoaderModelOnly, engine="sgld" uses SGLDUNETLoader/SGLDOptions/
SGLDLoraLoader; everything else is identical so A/C pairs differ only in the model path.
"""

import copy

SIGMAS_20 = (
    "1.0, 0.9956331849098206, 0.9908256530761719, 0.9855073094367981, 0.9795918464660645, "
    "0.9729729890823364, 0.9655172228813171, 0.9570552110671997, 0.9473683834075928, "
    "0.936170220375061, 0.9230769276618958, 0.9075629711151123, 0.8888888955116272, "
    "0.8659793734550476, 0.8372092843055725, 0.800000011920929, 0.75, 0.6792452931404114, "
    "0.5714285969734192, 0.3870967924594879, 0.0"
)
SIGMAS_8 = (
    "1.0, 0.9882352948188782, 0.9729729890823364, 0.9523809552192688, 0.9230769276618958, "
    "0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0"
)

PROMPTS = {
    "cat": "A small orange cat walks slowly on a sunny beach. Gentle ocean waves. Audio: quiet waves and a soft piano melody, no speech.",
    "speech": "A woman in a floral dress smiles and says to a friend: \"It is good to see you again, it has been a long time.\" Indoor living room, warm light. Audio: clear female voice, quiet room tone.",
    "music": "Close-up of a street musician playing an acoustic guitar at dusk, fingers moving on the strings. Audio: a lively fingerstyle guitar tune, light city ambience.",
    "motion": "A red sports car drifts around a wet mountain corner at night, the camera tracks fast alongside it, water spraying from the tires. Audio: engine roar and tire screech.",
    "multi": "Two children and a golden retriever play in a park, a sign that reads OPEN DAILY in the background, the camera slowly pans right. Audio: laughter and a dog barking.",
    "ref": "Use <Picture 1> as the cat reference. The orange cat walks on a sunny beach. Audio: quiet ocean waves and soft piano, no speech.",
}

DEFAULT = {
    "engine": "sgld",
    "task": "t2v",  # t2v | fl2v | r2v
    "unet": "minimax_h3_fl2va_int8_convrot.safetensors",
    "weight_dtype": "default",
    "sgld": {"model_type": "minimax_h3", "attention_backend": "torch_sdpa", "dit_layerwise_offload": True},
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "clip_device": "default",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
    "avae": "minimax_h3_audio_vae_fp32.safetensors",
    "width": 864,
    "height": 480,
    "length": 107,
    "resolution_selector": None,  # (aspect_ratio, megapixels)
    "prompt": PROMPTS["cat"],
    "first": None,
    "last": None,
    "first_scale_mp": None,
    "refs": [],
    "ref_videos": [],
    "ref_audios": [],
    "ref_image_size": "match",
    "guides": [],  # [(frame_idx, image)]
    "sampler": "euler",
    "sigmas": ("manual", SIGMAS_20),  # ("manual", str) | ("basic", sched, steps, denoise) | ("distilled", kind)
    "loras": [],  # [(name, strength, nickname)]
    "cfg": None,  # (cfg, negative_prompt)
    "sigma_shift": None,  # (video, audio)
    "block_sparse": None,  # dict(selection=..., keep_percent=...)
    "model_attention": None,  # "comfy kitchen attention" | ...
    "controlnet": None,  # dict(patch=..., video=..., strength=...)
    "denoise_mask": None,  # image file whose alpha drives LoadImage's MASK
    "seed": 150000,
    "batch2": False,
    "latent_capture": True,
    "profile": False,  # H3BenchmarkModel: CUDA-synced sampler and per-DiT-call timings
    "save": True,
    "save_prefix": "video/campaign/run",
}


class _G:
    def __init__(self):
        self.g, self.n = {}, 0

    def add(self, cls, **inputs):
        self.n += 1
        k = str(self.n)
        self.g[k] = {"class_type": cls, "inputs": inputs}
        return k


def build(cfg_in):
    cfg = copy.deepcopy(DEFAULT)
    for key, value in cfg_in.items():
        if key == "sgld":
            cfg["sgld"] = {**cfg["sgld"], **value}
        else:
            cfg[key] = value
    G = _G()
    sgld = cfg["engine"] == "sgld"

    if sgld:
        opts = G.add("SGLDOptions", **cfg["sgld"])
        model = G.add("SGLDUNETLoader", unet_name=cfg["unet"], weight_dtype=cfg["weight_dtype"], sgld_options=[opts, 0])
    else:
        model = G.add("UNETLoader", unet_name=cfg["unet"], weight_dtype=cfg["weight_dtype"])
    model_out = [model, 0]
    for name, strength, nick in cfg["loras"]:
        if sgld:
            k = G.add("SGLDLoraLoader", model=model_out, lora_name=name, strength_model=strength, nickname=nick, target="all")
        else:
            k = G.add("LoraLoaderModelOnly", model=model_out, lora_name=name, strength_model=strength)
        model_out = [k, 0]
    if cfg["sigma_shift"]:
        sv, sa = cfg["sigma_shift"]
        model_out = [G.add("MiniMaxH3SigmaShift", model=model_out, shift_video=sv, shift_audio=sa), 0]
    if cfg["block_sparse"]:
        b = cfg["block_sparse"]
        model_out = [G.add("BlockSparseAttention", model=model_out, selection=b.get("selection", "vsa"),
                           **{"selection.keep_percent": b.get("keep_percent", 10.0)},
                           start_percent=b.get("start_percent", 0.2), end_percent=b.get("end_percent", 1.0),
                           dense_blocks=b.get("dense_blocks", ""), min_tokens=b.get("min_tokens", 12288),
                           extra_tokens=b.get("extra_tokens", 256), sink_conditioning=b.get("sink", "exact_kv_and_rows"),
                           verbose=False), 0]
    if cfg["model_attention"]:
        model_out = [G.add("ModelAttentionBackend", model=model_out, attention=cfg["model_attention"]), 0]

    clip = G.add("CLIPLoader", clip_name=cfg["clip"], type="minimax", device=cfg["clip_device"])
    vae = G.add("VAELoader", vae_name=cfg["vae"])
    avae = G.add("VAELoader", vae_name=cfg["avae"])

    if cfg["controlnet"]:
        c = cfg["controlnet"]
        patch = G.add("ModelPatchLoader", name=c["patch"])
        vid = G.add("LoadVideo", file=c["video"])
        comps = G.add("GetVideoComponents", video=[vid, 0])
        model_out = [G.add("MiniMaxH3FunControlNetApply", model=model_out, model_patch=[patch, 0], vae=[vae, 0],
                           strength=c.get("strength", 1.0), start_percent=0.0, end_percent=1.0,
                           control_video=[comps, 0]), 0]
    if cfg["profile"]:
        model_out = [G.add("H3BenchmarkModel", model=model_out, run_id="set_by_runner"), 0]

    if cfg["resolution_selector"]:
        aspect, mp = cfg["resolution_selector"]
        rs = G.add("ResolutionSelector", aspect_ratio=aspect, megapixels=mp, multiple=32)
        width, height = [rs, 0], [rs, 1]
    else:
        width, height = cfg["width"], cfg["height"]

    if cfg["task"] == "r2v":
        inputs = dict(clip=[clip, 0], prompt=cfg["prompt"], width=width, height=height, length=cfg["length"],
                      ref_image_size=cfg["ref_image_size"], vae=[vae, 0], audio_vae=[avae, 0])
        for i, img in enumerate(cfg["refs"]):
            inputs[f"ref_images.ref_image_{i}"] = [G.add("LoadImage", image=img), 0]
        for i, v in enumerate(cfg["ref_videos"]):
            comps = G.add("GetVideoComponents", video=[G.add("LoadVideo", file=v), 0])
            inputs[f"ref_videos.ref_video_{i}"] = [comps, 0]
            inputs[f"ref_video_audios.ref_video_audio_{i}"] = [comps, 1]
        for i, a in enumerate(cfg["ref_audios"]):
            inputs[f"ref_audios.ref_audio_{i}"] = [G.add("LoadAudio", audio=a), 0]
        cond = G.add("MiniMaxH3ReferenceToVideo", **inputs)
    else:
        inputs = dict(clip=[clip, 0], vae=[vae, 0], prompt=cfg["prompt"], width=width, height=height, length=cfg["length"])
        for key in ("first", "last"):
            if cfg[key]:
                img = [G.add("LoadImage", image=cfg[key]), 0]
                if cfg["first_scale_mp"]:
                    img = [G.add("ImageScaleToTotalPixels", image=img, upscale_method="nearest-exact",
                                 megapixels=cfg["first_scale_mp"], resolution_steps=32), 0]
                inputs[f"{key}_frame"] = img
        cond = G.add("MiniMaxH3ImageToVideo", **inputs)
    positive, latent = [cond, 0], [cond, 1]
    for frame_idx, img in cfg["guides"]:
        positive = [G.add("MiniMaxH3AddGuide", positive=positive, latent=latent, frame_idx=frame_idx,
                          vae=[vae, 0], audio_vae=[avae, 0], image=[G.add("LoadImage", image=img), 0]), 0]

    if cfg["denoise_mask"]:
        m = G.add("LoadImage", image=cfg["denoise_mask"])
        latent = [G.add("SetLatentNoiseMask", samples=latent, mask=[m, 1]), 0]
    if cfg["batch2"]:
        sep = G.add("LTXVSeparateAVLatent", av_latent=latent)
        v2 = G.add("RepeatLatentBatch", samples=[sep, 0], amount=2)
        a2 = G.add("RepeatLatentBatch", samples=[sep, 1], amount=2)
        latent = [G.add("LTXVConcatAVLatent", video_latent=[v2, 0], audio_latent=[a2, 0]), 0]

    kind = cfg["sigmas"][0]
    if kind == "manual":
        sigmas = G.add("ManualSigmas", sigmas=cfg["sigmas"][1])
    elif kind == "basic":
        _, sched, steps, denoise = cfg["sigmas"]
        sigmas = G.add("BasicScheduler", model=model_out, scheduler=sched, steps=steps, denoise=denoise)
    elif kind == "distilled":
        sigmas = G.add("SGLDH3DistilledSigmas", model_type=cfg["sigmas"][1])
    else:
        raise ValueError(kind)
    sampler = G.add("KSamplerSelect", sampler_name=cfg["sampler"])
    noise = G.add("RandomNoise", noise_seed=cfg["seed"])
    if cfg["cfg"]:
        scale, neg = cfg["cfg"]
        # The negative needs the same H3 payload (layout, token tags) as the positive.
        negc = G.add("MiniMaxH3ImageToVideo", clip=[clip, 0], vae=[vae, 0], prompt=neg,
                     width=width, height=height, length=cfg["length"])
        guider = G.add("CFGGuider", model=model_out, positive=positive, negative=[negc, 0], cfg=scale)
    else:
        guider = G.add("BasicGuider", model=model_out, conditioning=positive)
    out = G.add("SamplerCustomAdvanced", noise=[noise, 0], guider=[guider, 0], sampler=[sampler, 0],
                sigmas=[sigmas, 0], latent_image=latent)
    final = [out, 0]
    if cfg["latent_capture"]:
        final = [G.add("H3BenchmarkLatent", latent=final, run_id="set_by_runner"), 0]
    if cfg["save"]:
        images = G.add("VAEDecode", samples=final, vae=[vae, 0])
        audio = G.add("VAEDecodeAudio", samples=final, vae=[avae, 0])
        video = G.add("CreateVideo", images=[images, 0], audio=[audio, 0], fps=24.0)
        G.add("SaveVideo", video=[video, 0], filename_prefix=cfg["save_prefix"], format="auto", codec="auto")
    return G.g, cfg
