"""Build ComfyUI API prompts for LTX-2.5 T2V (official template structure and a minimal one).

The official template ``video_ltx2_5_t2v.json`` is a subgraph; this flattens it with
the prompt enhancer off (its default), so the only difference between the two
variants is UNETLoader vs SGLDUNETLoader (+ SGLDOptions).
"""

import json

UNET = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
CLIP = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"
PROMPT = (
    "Dynamic cinematic close-up of high-tech modular machinery self-assembling in "
    "midair, precision robotic parts, magnetic connectors, and glowing circuits "
    "clicking together, subtle smoke and light flares, extremely detailed titanium "
    "textures. The final product displays a clean, clear surface with large glowing "
    "engraved text “LTX-2.5” centered and unobstructed, dramatic lighting, "
    "photorealism, 8K, sharp focus.\n"
)
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"


def _model_nodes(sgld: bool, sgld_options: dict | None):
    if not sgld:
        return {"384": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}}}
    nodes = {
        "384": {
            "class_type": "SGLDUNETLoader",
            "inputs": {"unet_name": UNET, "weight_dtype": "default", "sgld_options": ["900", 0]},
        },
        "900": {"class_type": "SGLDOptions", "inputs": {"model_type": "auto-detect", **(sgld_options or {})}},
    }
    return nodes


def official_t2v(
    *,
    sgld: bool,
    prefix: str,
    width: int = 1280,
    height: int = 720,
    seconds: int = 5,
    fps: int = 24,
    seed: int = 558811532553686,
    prompt: str = PROMPT,
    sgld_options: dict | None = None,
    two_stage: bool = True,
) -> dict:
    frames = seconds * fps + 1
    p = {
        **_model_nodes(sgld, sgld_options),
        "385": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "386": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "387": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "ltxv", "device": "default"}},
        "364": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["387", 0], "text": prompt}},
        "373": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["387", 0], "text": NEGATIVE}},
        "365": {"class_type": "LTXVConditioning", "inputs": {"positive": ["364", 0], "negative": ["373", 0], "frame_rate": float(fps)}},
        "356": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width // 2 if two_stage else width, "height": height // 2 if two_stage else height, "length": frames, "batch_size": 1}},
        "366": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["386", 0], "frames_number": frames, "frame_rate": fps, "batch_size": 1}},
        "377": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["356", 0], "audio_latent": ["366", 0]}},
        "339": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "352": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "404": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE1_SIGMAS}},
        "388": {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["384", 0], "positive": ["365", 0], "negative": ["365", 1], "video_cfg": 1.0, "audio_cfg": 1.0}},
        "344": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["339", 0], "guider": ["388", 0], "sampler": ["352", 0], "sigmas": ["404", 0], "latent_image": ["377", 0]}},
        "367": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["344", 0]}},
    }
    final = "367"
    if two_stage:
        p.update({
            "371": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALER}},
            "348": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["367", 0], "upscale_model": ["371", 0], "vae": ["385", 0]}},
            "340": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["348", 0], "audio_latent": ["367", 1]}},
            "338": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
            "341": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
            "395": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE2_SIGMAS}},
            "391": {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["384", 0], "positive": ["365", 0], "negative": ["365", 1], "video_cfg": 1.0, "audio_cfg": 1.0}},
            "368": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["338", 0], "guider": ["391", 0], "sampler": ["341", 0], "sigmas": ["395", 0], "latent_image": ["340", 0]}},
            "369": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["368", 0]}},
        })
        final = "369"
    p.update({
        "374": {"class_type": "VAEDecodeTiled", "inputs": {"samples": [final, 0], "vae": ["385", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}},
        "358": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": [final, 1], "audio_vae": ["386", 0]}},
        "370": {"class_type": "CreateVideo", "inputs": {"images": ["374", 0], "audio": ["358", 0], "fps": float(fps)}},
        "75": {"class_type": "SaveVideo", "inputs": {"video": ["370", 0], "filename_prefix": f"{prefix}/video", "format": "mp4", "codec": "h264"}},
        "910": {"class_type": "SaveLatent", "inputs": {"samples": [final, 0], "filename_prefix": f"{prefix}/video_latent"}},
        "911": {"class_type": "SaveLatent", "inputs": {"samples": [final, 1], "filename_prefix": f"{prefix}/audio_latent"}},
    })
    return p


def dual_cfg(p: dict, video_cfg: float, audio_cfg: float) -> dict:
    for nid in ("388", "391"):
        if nid in p:
            p[nid]["inputs"]["video_cfg"] = video_cfg
            p[nid]["inputs"]["audio_cfg"] = audio_cfg
    return p


if __name__ == "__main__":
    print(json.dumps(official_t2v(sgld=True, prefix="x"), indent=1))


I2V_PROMPT = (
    "Use the provided start image as the first frame. The machine slowly rotates as "
    "glowing circuits pulse along its surface, small robotic arms click into place and "
    "sparks drift through the smoky air. The camera pushes in toward the glowing "
    "engraved text. Low mechanical hum, metallic clicks."
)


def official_i2v(*, sgld: bool, prefix: str, image: str, seed: int = 875362541677469,
                 width: int = 1280, height: int = 720, seconds: int = 5, fps: int = 24,
                 prompt: str = I2V_PROMPT, sgld_options: dict | None = None) -> dict:
    """Flattened ``video_ltx2_5_i2v.json`` (prompt enhancer off)."""
    p = official_t2v(sgld=sgld, prefix=prefix, width=width, height=height, seconds=seconds,
                     fps=fps, seed=seed, prompt=prompt, sgld_options=sgld_options)
    p.update({
        "920": {"class_type": "LoadImage", "inputs": {"image": image}},
        "351": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["920", 0], "resize_type": "scale longer dimension", "resize_type.longer_size": 1536, "scale_method": "lanczos"}},
        "350": {"class_type": "LTXVPreprocess", "inputs": {"image": ["351", 0], "img_compression": 18}},
        "357": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["385", 0], "image": ["350", 0], "latent": ["356", 0], "strength": 0.7, "bypass": False}},
        "349": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["385", 0], "image": ["350", 0], "latent": ["348", 0], "strength": 1.0, "bypass": False}},
    })
    p["377"]["inputs"]["video_latent"] = ["357", 0]
    p["340"]["inputs"]["video_latent"] = ["349", 0]
    return p


LTX23_CKPT = "ltx-2.3-22b-distilled-fp8.safetensors"
LTX23_TE = "gemma_3_12B_it_fp8_scaled.safetensors"
LTX23_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def ltx23_t2v(*, sgld: bool, prefix: str, width: int = 1280, height: int = 720,
              seconds: int = 5, fps: int = 25, seed: int = 810138461690240,
              prompt: str = PROMPT.replace("LTX-2.5", "LTX-2.3"),
              sgld_options: dict | None = None) -> dict:
    """Flattened ``video_ltx2_3_t2v.json`` (enhancer off, bypassed image nodes dropped).

    Uses the distilled FP8 all-in-one checkpoint instead of dev-FP8 + distilled
    LoRA, so the LoraLoaderModelOnly node is omitted. CheckpointLoaderSimple
    still supplies the VAE; with SGLD its MODEL output is left unconnected.
    """
    frames = seconds * fps + 1
    model = ["236", 0]
    p = {
        "236": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": LTX23_CKPT}},
        "243": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": LTX23_TE, "ckpt_name": LTX23_CKPT, "device": "default"}},
        "221": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": LTX23_CKPT}},
        "240": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": prompt}},
        "247": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": NEGATIVE}},
        "239": {"class_type": "LTXVConditioning", "inputs": {"positive": ["240", 0], "negative": ["247", 0], "frame_rate": float(fps)}},
        "228": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width // 2, "height": height // 2, "length": frames, "batch_size": 1}},
        "214": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["221", 0], "frames_number": frames, "frame_rate": fps, "batch_size": 1}},
        "222": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["228", 0], "audio_latent": ["214", 0]}},
        "237": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "209": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "252": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE1_SIGMAS}},
        "231": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["239", 0], "negative": ["239", 1], "cfg": 1.0}},
        "215": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["237", 0], "guider": ["231", 0], "sampler": ["209", 0], "sigmas": ["252", 0], "latent_image": ["222", 0]}},
        "217": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["215", 0]}},
        "233": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LTX23_UPSCALER}},
        "253": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["217", 0], "upscale_model": ["233", 0], "vae": ["236", 2]}},
        "229": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["253", 0], "audio_latent": ["217", 1]}},
        "212": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["239", 0], "negative": ["239", 1], "latent": ["217", 0]}},
        "213": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["212", 0], "negative": ["212", 1], "cfg": 1.0}},
        "216": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "246": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "211": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE2_SIGMAS}},
        "219": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["216", 0], "guider": ["213", 0], "sampler": ["246", 0], "sigmas": ["211", 0], "latent_image": ["229", 0]}},
        "218": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["219", 0]}},
        "251": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["218", 0], "vae": ["236", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4}},
        "220": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["218", 1], "audio_vae": ["221", 0]}},
        "242": {"class_type": "CreateVideo", "inputs": {"images": ["251", 0], "audio": ["220", 0], "fps": float(fps)}},
        "75": {"class_type": "SaveVideo", "inputs": {"video": ["242", 0], "filename_prefix": f"{prefix}/video", "format": "mp4", "codec": "h264"}},
        "910": {"class_type": "SaveLatent", "inputs": {"samples": ["218", 0], "filename_prefix": f"{prefix}/video_latent"}},
        "911": {"class_type": "SaveLatent", "inputs": {"samples": ["218", 1], "filename_prefix": f"{prefix}/audio_latent"}},
    }
    if sgld:
        p["384"] = {"class_type": "SGLDUNETLoader", "inputs": {"unet_name": LTX23_CKPT, "weight_dtype": "default", "sgld_options": ["900", 0]}}
        p["900"] = {"class_type": "SGLDOptions", "inputs": {"model_type": "auto-detect", **(sgld_options or {})}}
        for nid in ("231", "213"):
            p[nid]["inputs"]["model"] = ["384", 0]
    return p


LTX23_DEV_CKPT = "ltx-2.3-22b-dev-fp8.safetensors"
LTX23_DISTILLED_LORA = "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
LTX23_FP4_TE = "gemma_3_12B_it_fp4_mixed.safetensors"
GEMMA_ABLITERATED_LORA = "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
LTX23_TEMPLATE_PROMPT = PROMPT.replace("LTX-2.5", "LTX-2.3")


def ltx23_template(*, sgld: bool, prefix: str, seed: int = 810138461690240,
                   enhance: bool = True, lora: bool = True, lora_strength: float = 0.5,
                   width: int = 1280, height: int = 720, seconds: int = 5, fps: int = 25,
                   prompt: str = LTX23_TEMPLATE_PROMPT, sgld_options: dict | None = None,
                   save_latents: bool = True, native_lora_on_sgld: bool = False) -> dict:
    """``video_ltx2_3_t2v.json`` exactly as shipped, flattened to API form.

    dev-FP8 + distilled LoRA 0.5, Gemma3 fp4_mixed + abliterated Gemma LoRA
    feeding the prompt enhancer, bypassed image-conditioning nodes kept.
    SGLD variant: SGLDUNETLoader + SGLDLoraLoader replace the MODEL chain; the
    LoraLoader still receives the SGLD MODEL (its MODEL output is unused).
    """
    frames = seconds * fps + 1
    model = ["232", 0]
    p = {
        "236": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": LTX23_DEV_CKPT}},
        "243": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": LTX23_FP4_TE, "ckpt_name": LTX23_DEV_CKPT, "device": "default"}},
        "326": {"class_type": "LoraLoader", "inputs": {"model": model, "clip": ["243", 0], "lora_name": GEMMA_ABLITERATED_LORA, "strength_model": 1.0, "strength_clip": 1.0}},
        "266": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": prompt}},
        "327": {"class_type": "TextGenerateLTX2Prompt", "inputs": {
            "clip": ["326", 1], "prompt": ["266", 0], "max_length": 2048,
            "sampling_mode": "on", "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64,
            "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
            "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": 0,
            "sampling_mode.presence_penalty": 0.0, "thinking": False, "use_default_template": True}},
        "330": {"class_type": "PrimitiveBoolean", "inputs": {"value": enhance}},
        "329": {"class_type": "ComfySwitchNode", "inputs": {"switch": ["330", 0], "on_false": ["266", 0], "on_true": ["327", 0]}},
        "328": {"class_type": "PreviewAny", "inputs": {"source": ["329", 0]}},
        "240": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": ["329", 0]}},
        "247": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["243", 0], "text": NEGATIVE}},
        "239": {"class_type": "LTXVConditioning", "inputs": {"positive": ["240", 0], "negative": ["247", 0], "frame_rate": float(fps)}},
        "201": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        "325": {"class_type": "EmptyImage", "inputs": {"width": 512, "height": 512, "batch_size": 1, "color": 0}},
        "238": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["325", 0], "resize_type": "scale dimensions", "resize_type.width": width, "resize_type.height": height, "resize_type.crop": "center", "scale_method": "lanczos"}},
        "331": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["238", 0], "resize_type": "scale longer dimension", "resize_type.longer_size": 512, "scale_method": "lanczos"}},
        "248": {"class_type": "LTXVPreprocess", "inputs": {"image": ["331", 0], "img_compression": 18}},
        "228": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width // 2, "height": height // 2, "length": frames, "batch_size": 1}},
        "249": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["236", 2], "image": ["248", 0], "latent": ["228", 0], "strength": 0.7, "bypass": ["201", 0]}},
        "221": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": LTX23_DEV_CKPT}},
        "214": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"audio_vae": ["221", 0], "frames_number": frames, "frame_rate": fps, "batch_size": 1}},
        "222": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["249", 0], "audio_latent": ["214", 0]}},
        "237": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "209": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "252": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE1_SIGMAS}},
        "231": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["239", 0], "negative": ["239", 1], "cfg": 1.0}},
        "215": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["237", 0], "guider": ["231", 0], "sampler": ["209", 0], "sigmas": ["252", 0], "latent_image": ["222", 0]}},
        "217": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["215", 0]}},
        "233": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LTX23_UPSCALER}},
        "253": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["217", 0], "upscale_model": ["233", 0], "vae": ["236", 2]}},
        "230": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["236", 2], "image": ["248", 0], "latent": ["253", 0], "strength": 1.0, "bypass": ["201", 0]}},
        "229": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["230", 0], "audio_latent": ["217", 1]}},
        "212": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["239", 0], "negative": ["239", 1], "latent": ["217", 0]}},
        "213": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": ["212", 0], "negative": ["212", 1], "cfg": 1.0}},
        "216": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "246": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "211": {"class_type": "ManualSigmas", "inputs": {"sigmas": STAGE2_SIGMAS}},
        "219": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["216", 0], "guider": ["213", 0], "sampler": ["246", 0], "sigmas": ["211", 0], "latent_image": ["229", 0]}},
        "218": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["219", 0]}},
        "251": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["218", 0], "vae": ["236", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4}},
        "220": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["218", 1], "audio_vae": ["221", 0]}},
        "242": {"class_type": "CreateVideo", "inputs": {"images": ["251", 0], "audio": ["220", 0], "fps": float(fps)}},
        "75": {"class_type": "SaveVideo", "inputs": {"video": ["242", 0], "filename_prefix": f"{prefix}/video", "format": "mp4", "codec": "h264"}},
    }
    if save_latents:
        p["910"] = {"class_type": "SaveLatent", "inputs": {"samples": ["218", 0], "filename_prefix": f"{prefix}/video_latent"}}
        p["911"] = {"class_type": "SaveLatent", "inputs": {"samples": ["218", 1], "filename_prefix": f"{prefix}/audio_latent"}}
    if sgld:
        p["384"] = {"class_type": "SGLDUNETLoader", "inputs": {"unet_name": LTX23_DEV_CKPT, "weight_dtype": "default", "sgld_options": ["900", 0]}}
        p["900"] = {"class_type": "SGLDOptions", "inputs": {"model_type": "auto-detect", **(sgld_options or {})}}
        p["232"] = {"class_type": "SGLDLoraLoader", "inputs": {"model": ["384", 0], "lora_name": LTX23_DISTILLED_LORA, "strength_model": lora_strength, "nickname": "ltx23_distilled", "target": "all"}}
    else:
        p["232"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["236", 0], "lora_name": LTX23_DISTILLED_LORA, "strength_model": lora_strength}}
    if native_lora_on_sgld:
        p["232"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["384", 0], "lora_name": LTX23_DISTILLED_LORA, "strength_model": lora_strength}}
    if not lora:
        base = ["384", 0] if sgld else ["236", 0]
        for nid in ("231", "213", "326"):
            p[nid]["inputs"]["model"] = base
        del p["232"]
    return p
