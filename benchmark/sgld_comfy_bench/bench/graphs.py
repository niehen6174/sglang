"""ComfyUI API graphs for H3, LTX-2.5 and Qwen-Image 2.1.

engine="stock" uses UNETLoader; engine="sgld" uses SGLDUNETLoader + SGLDOptions. Nothing else in
a graph depends on the engine, so a stock/SGLD pair differs only in who runs the DiT.
Every graph ends in H3BenchmarkLatent (final latent + sha256) and a Save node.
"""
import copy

import h3graph

H3_PROMPT = h3graph.PROMPTS["cat"]
# W-H3-ref4 / W-H3-ref8: one tag per reference image (inputs/ref_01.jpg .. ref_08.jpg).
REF_SUBJECTS = ["the orange cat", "the golden retriever", "the woman in a red coat", "the old man with a hat",
                "the blue vintage car", "the acoustic guitar", "the small wooden house", "the silver robot"]


def ref_prompt(n):
    tags = ", ".join(f"<Picture {i + 1}> is {REF_SUBJECTS[i]}" for i in range(n))
    return (f"{tags}. All of them appear together in one sunny park scene, the camera slowly pans right "
            "past each of them. Audio: birds, light wind and soft piano, no speech.")


def ref_images(n):
    return [f"ref_{i + 1:02d}.jpg" for i in range(n)]


def h3_sigmas(steps, shift=12.0):
    """H3 video schedule: t linear in [1, 0], sigma = shift*t / (1 + (shift-1)*t)."""
    vals = []
    for i in range(steps + 1):
        t = 1.0 - i / steps
        vals.append(shift * t / (1.0 + (shift - 1.0) * t))
    return ", ".join(f"{v:.10f}" for v in vals)


def h3(engine, files, sgld, *, task="t2v", width=1344, height=768, length=124, steps=20, distilled=None,
       refs=0, seed=150000, prefix="video/bench/h3"):
    cfg = dict(engine="sgld" if engine == "sgld" else "official", task=task, unet=files["ref"] if task == "r2v" else files["dit"],
               clip=files["clip"], vae=files["vae"], avae=files["avae"], width=width, height=height, length=length,
               prompt=H3_PROMPT, seed=seed, save_prefix=prefix, profile=True,
               sigmas=("distilled", distilled) if distilled else ("manual", h3_sigmas(steps)))
    if task == "r2v":
        cfg.update(prompt=ref_prompt(refs), refs=ref_images(refs), ref_image_size="match")
    if engine == "sgld":
        cfg["sgld"] = sgld
    if files.get("unet_override"):
        cfg["unet"] = files["unet_override"]
    graph, _ = h3graph.build(cfg)
    if engine == "sgld":
        # h3graph merges into its own defaults; reset to exactly the requested options.
        for node in graph.values():
            if node["class_type"] == "SGLDOptions":
                node["inputs"] = dict(sgld)
    return graph


class _G:
    def __init__(self):
        self.g, self.n = {}, 0

    def add(self, cls, **inputs):
        self.n += 1
        self.g[str(self.n)] = {"class_type": cls, "inputs": inputs}
        return str(self.n)


def _model(G, engine, unet, sgld):
    """Loader plus H3BenchmarkModel, which records CUDA-synced sampler and per-DiT-call seconds."""
    if engine == "sgld":
        opts = G.add("SGLDOptions", **sgld)
        model = G.add("SGLDUNETLoader", unet_name=unet, weight_dtype="default", sgld_options=[opts, 0])
    else:
        model = G.add("UNETLoader", unet_name=unet, weight_dtype="default")
    return [G.add("H3BenchmarkModel", model=[model, 0], run_id="set_by_runner"), 0]


LTX_PROMPT = ("Dynamic cinematic close-up of high-tech modular machinery self-assembling in midair, precision "
              "robotic parts, magnetic connectors, and glowing circuits clicking together, subtle smoke and light "
              "flares, extremely detailed, camera slowly orbits. Audio: mechanical clicks and a low synth hum.")
LTX_NEG = "pc game, console game, video game, cartoon, childish, ugly"
LTX_STAGE1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_STAGE2 = "0.85, 0.7250, 0.4219, 0.0"


def ltx(engine, files, sgld, *, width=640, height=360, length=121, fps=24, two_stage=True, seed=4242,
        unet=None, prefix="video/bench/ltx"):
    """The official LTX-2.5 distilled T2V template: stage 1 at width x height, x2 latent upsample,
    stage 2 at 2x the size. Both stages share the loaded DiT."""
    G = _G()
    model = _model(G, engine, unet or files["dit"], sgld)
    vae = G.add("VAELoader", vae_name=files["vae"])
    avae = G.add("VAELoader", vae_name=files["avae"])
    clip = G.add("CLIPLoader", clip_name=files["clip"], type="ltxv", device="default")
    pos = G.add("CLIPTextEncode", clip=[clip, 0], text=LTX_PROMPT)
    neg = G.add("CLIPTextEncode", clip=[clip, 0], text=LTX_NEG)
    cond = G.add("LTXVConditioning", positive=[pos, 0], negative=[neg, 0], frame_rate=float(fps))
    vlat = G.add("EmptyLTXVLatentVideo", width=width, height=height, length=length, batch_size=1)
    alat = G.add("LTXVEmptyLatentAudio", audio_vae=[avae, 0], frames_number=length, frame_rate=fps, batch_size=1)
    av = G.add("LTXVConcatAVLatent", video_latent=[vlat, 0], audio_latent=[alat, 0])

    def stage(latent, sigmas, noise_seed):
        guider = G.add("LTXVDualCFGGuider", model=model, positive=[cond, 0], negative=[cond, 1],
                       video_cfg=1.0, audio_cfg=1.0)
        return G.add("SamplerCustomAdvanced", noise=[G.add("RandomNoise", noise_seed=noise_seed), 0],
                     guider=[guider, 0], sampler=[G.add("KSamplerSelect", sampler_name="euler_ancestral"), 0],
                     sigmas=[G.add("ManualSigmas", sigmas=sigmas), 0], latent_image=latent)

    out = stage([av, 0], LTX_STAGE1, seed)
    if two_stage:
        sep = G.add("LTXVSeparateAVLatent", av_latent=[out, 0])
        up_model = G.add("LatentUpscaleModelLoader", model_name=files["upscaler"])
        up = G.add("LTXVLatentUpsampler", samples=[sep, 0], upscale_model=[up_model, 0], vae=[vae, 0])
        av2 = G.add("LTXVConcatAVLatent", video_latent=[up, 0], audio_latent=[sep, 1])
        out = stage([av2, 0], LTX_STAGE2, seed + 1)
    final = G.add("H3BenchmarkLatent", latent=[out, 0], run_id="set_by_runner")
    sep = G.add("LTXVSeparateAVLatent", av_latent=[final, 0])
    frames = G.add("VAEDecodeTiled", samples=[sep, 0], vae=[vae, 0], tile_size=512, overlap=64,
                   temporal_size=64, temporal_overlap=16)
    audio = G.add("LTXVAudioVAEDecode", samples=[sep, 1], audio_vae=[avae, 0])
    video = G.add("CreateVideo", images=[frames, 0], audio=[audio, 0], fps=float(fps))
    G.add("SaveVideo", video=[video, 0], filename_prefix=prefix, format="mp4", codec="h264")
    return G.g


QWEN_PROMPT = ("Greyscale fashion editorial portrait of an avant-garde woman tilted upwards in profile, striking "
               "high-contrast chiaroscuro lighting, oversized thick-rimmed circular black sunglasses, sculptural "
               "black coat, studio backdrop, medium format film grain.")
QWEN_EDIT_PROMPT = "Turn the cat into a watercolor painting, keep the pose."


def qwen(engine, files, sgld, *, width=1024, height=1024, steps=20, cfg=1.0, edit=False, seed=1234,
         unet=None, prefix="image/bench/qwen"):
    G = _G()
    model = _model(G, engine, unet or files["dit"], sgld)
    clip = G.add("CLIPLoader", clip_name=files["clip"], type="qwen_image", device="default")
    vae = G.add("VAELoader", vae_name=files["vae"])
    enc = dict(clip=[clip, 0], prompt=QWEN_EDIT_PROMPT if edit else QWEN_PROMPT,
               negative_prompt="" if cfg == 1.0 else "blurry, low quality, distorted", resolution=1024)
    if edit:
        enc.update(vae=[vae, 0], **{"images.image_1": [G.add("LoadImage", image="cmp_cat_small.png"), 0]})
    text = G.add("TextEncodeQwenImage21", **enc)
    latent = G.add("EmptyLatentImage", width=width, height=height, batch_size=1)
    out = G.add("KSampler", model=model, seed=seed, steps=steps, cfg=cfg, sampler_name="euler", scheduler="simple",
                positive=[text, 0], negative=[text, 1], latent_image=[latent, 0], denoise=1.0)
    final = G.add("H3BenchmarkLatent", latent=[out, 0], run_id="set_by_runner")
    img = G.add("VAEDecode", samples=[final, 0], vae=[vae, 0])
    G.add("SaveImage", images=[img, 0], filename_prefix=prefix)
    return G.g


def with_steps(graph, model, steps):
    """Warmup copy of a graph with fewer denoising steps at the same shape (kernels, allocations and
    any per-shape capture are the same; only the step count differs)."""
    g = copy.deepcopy(graph)
    for node in g.values():
        ct, inp = node["class_type"], node["inputs"]
        if model == "h3" and ct == "ManualSigmas":
            inp["sigmas"] = h3_sigmas(steps)
        elif model == "qwen" and ct == "KSampler":
            inp["steps"] = steps
    return g


def set_seed(graph, seed):
    """Seed every sampler; LTX's second stage gets seed + 1 as in the template."""
    k = 0
    for node in graph.values():
        ct, inp = node["class_type"], node["inputs"]
        if ct == "RandomNoise":
            inp["noise_seed"] = seed + k
            k += 1
        elif ct == "KSampler":
            inp["seed"] = seed
    return graph
