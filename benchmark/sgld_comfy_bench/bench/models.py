"""Model files per card, and what setup/download.py fetches.

Track A uses ComfyUI single files (Comfy-Org / Lightricks repos), linked into ComfyUI/models.
Track B uses the HF diffusers repos both engines load natively. Revisions are pinned so every
machine downloads the same bytes.
"""
import os

CARDS = ("4090", "5090", "h200", "b200")
BIG = ("h200", "b200")
SMALL = ("4090", "5090")

# ---- ComfyUI single files (track A) ------------------------------------------------------
H3_REPO, H3_REV = "Comfy-Org/MiniMax-H3", "e5eb578a89295337b8ff433a035929ce0279e0b6"
H3 = dict(
    fl_bf16="minimax_h3_fl2va_bf16.safetensors",                # 66.3 GB
    fl_int8="minimax_h3_fl2va_int8_convrot.safetensors",        # 34.0 GB
    fl_pruned_int8="minimax_h3_fl2va_pruned_int8_convrot.safetensors",    # 21.0 GB
    ref_int8="minimax_h3_ref2va_int8_convrot.safetensors",      # 34.0 GB
    ref_pruned_int8="minimax_h3_ref2va_pruned_int8_convrot.safetensors",  # 21.0 GB
    te_nvfp4="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",    # 15.7 GB
    te_int8="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",  # 27.1 GB
    vae_fp16="minimax_h3_video_vae_fp16.safetensors",           # 5.2 GB
    vae_int8="minimax_h3_video_vae_int8_convrot.safetensors",   # 2.8 GB
    avae="minimax_h3_audio_vae_fp32.safetensors",               # 0.6 GB
)
H3_FOLDER = {k: ("diffusion_models" if k.startswith(("fl_", "ref_")) else
                 "text_encoders" if k.startswith("te_") else "vae") for k in H3}

LTX_REPO, LTX_REV = "Lightricks/LTX-2.5", "49b093a39c19"
LTX = dict(
    dit_int8="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",  # 21.5 GB
    dit_bf16="ltx-2.5-22b-distilled-transformer-bf16.safetensors",                # 42.0 GB
    te_int8="gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",        # 15.4 GB
    vae="ltx-2.5-video-vae-bf16.safetensors",                                     # 1.5 GB
    avae="ltx-2.5-audio-vae-bf16.safetensors",                                    # 0.4 GB
    upscaler="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",           # 1.0 GB
)
LTX_FOLDER = dict(dit_int8="diffusion_models", dit_bf16="diffusion_models", te_int8="text_encoders",
                  vae="vae", avae="vae", upscaler="latent_upscale_models")

QWEN_REPO, QWEN_REV = "Comfy-Org/Qwen-Image-2.1", None
QWEN = dict(
    dit_bf16="qwen_image_2.1_bf16.safetensors",        # 14.2 GB
    dit_int8="qwen_image_2.1_int8_convrot.safetensors",  # 7.3 GB
    te="qwen3vl_8b_int8_convrot.safetensors",          # 9.4 GB
    vae="qwen_image_2.1_vae_bf16.safetensors",         # 0.7 GB
)
QWEN_FOLDER = dict(dit_bf16="diffusion_models", dit_int8="diffusion_models", te="text_encoders", vae="vae")

# Distilled H3 variants run by SGLD from a diffusers-layout directory (BF16, ~66 GB each).
FASTH3_REPO = "FastVideo/FastVideo-FastH3-8-Step-V2"
VDN_REPO, VDN_REV = "OpenVDN/vdn-minimax-h3", "50744d80bde8c1adbc5f37089f125e0e436bf7b0"


def ws(*parts):
    return os.path.join(os.environ["WORKSPACE"], *parts)


FASTH3_DIR = lambda: ws("models", "FastH3-8step-V2")  # noqa: E731
VDN_DIR = lambda: ws("models", "VDN-H3-8nfe")  # noqa: E731


def h3_files(card):
    """Card defaults for H3 in ComfyUI. 4090/5090 only fit the pruned INT8 DiT."""
    if card in BIG:
        f = dict(dit=H3["fl_bf16"], dit_int8=H3["fl_int8"], ref=H3["ref_int8"], clip=H3["te_nvfp4"], vae=H3["vae_fp16"])
    else:
        f = dict(dit=H3["fl_pruned_int8"], dit_int8=H3["fl_pruned_int8"], ref=H3["ref_pruned_int8"],
                 clip=H3["te_nvfp4"], vae=H3["vae_int8"])
    if os.environ.get("KIT_H3_CLIP"):
        f["clip"] = os.environ["KIT_H3_CLIP"]
    f["avae"] = H3["avae"]
    return f


def ltx_files(card):
    return dict(dit=LTX["dit_int8"], dit_bf16=LTX["dit_bf16"] if card in BIG else None,
                clip=LTX["te_int8"], vae=LTX["vae"], avae=LTX["avae"], upscaler=LTX["upscaler"])


def qwen_files(card):
    return dict(dit=QWEN["dit_int8"] if card == "4090" else QWEN["dit_bf16"], dit_int8=QWEN["dit_int8"],
                clip=QWEN["te"], vae=QWEN["vae"])


def track_a_downloads(card):
    """(repo, revision, path-in-repo, ComfyUI models subfolder)"""
    out = []
    h3_keys = ["te_nvfp4", "avae", "fl_pruned_int8", "ref_pruned_int8", "vae_int8"]
    if card in BIG:
        h3_keys += ["fl_bf16", "fl_int8", "ref_int8", "vae_fp16"]
    if card == "4090":
        h3_keys += ["te_int8"]  # fallback TE if NVFP4 is too slow on sm89 (KIT_H3_CLIP)
    for k in h3_keys:
        out.append((H3_REPO, H3_REV, f"{H3_FOLDER[k]}/{H3[k]}", H3_FOLDER[k]))
    for k in LTX:
        if k == "dit_bf16" and card not in BIG:
            continue
        out.append((LTX_REPO, LTX_REV, f"{LTX_FOLDER[k]}/{LTX[k]}", LTX_FOLDER[k]))
    for k in QWEN:
        out.append((QWEN_REPO, QWEN_REV, f"{QWEN_FOLDER[k]}/{QWEN[k]}", QWEN_FOLDER[k]))
    return out


# Functional matrix only (h200, 5090): H3 turbo LoRA (F-02), LTX-2.3 FP8 all-in-one + LoRA (F-07).
FUNCTIONAL_CARDS = ("h200", "5090")
FUNCTIONAL_DOWNLOADS = [
    (H3_REPO, H3_REV, "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "loras"),
    ("Lightricks/LTX-2.3-fp8", None, "ltx-2.3-22b-dev-fp8.safetensors", "checkpoints"),
    ("Lightricks/LTX-2.3-fp8", None, "ltx-2.3-22b-dev-fp8.safetensors", "diffusion_models"),
    ("Comfy-Org/ltx-2", None, "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", "text_encoders"),
    ("Comfy-Org/ltx-2.3", None, "split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
     "loras"),
]


# ---- HF diffusers repos (track B) --------------------------------------------------------
# (repo, revision, allow_patterns). Only the parts the benchmark loads are fetched.
TRACK_B_REPOS = [
    # The FL2VA / Ref2VA partitions (~143 GB each): SGLang's --model-variant and vllm-omni's
    # --task-type both load these, not the root-level components.
    ("MiniMaxAI/MiniMax-H3", "42ed227ee7df", ["*.json", "FL2VA/**", "Ref2VA/**", "audio_scheduler/**", "scheduler/**"]),
    # Distilled one-/two-stage: transformer, connectors, text encoder, VAEs, vocoder, upsampler (~120 GB).
    ("Lightricks/LTX-2.5-Diffusers", "da0b69096002",
     ["*.json", "*.txt", "tokenizer*/**", "scheduler/**", "transformer/**", "connectors/**", "text_encoder/**",
      "vae/**", "audio_vae/**", "vocoder/**", "latent_upsampler/**"]),
    ("Qwen/Qwen-Image-2.1", "d26bb61231c3", None),  # 33 GB
    # vllm-omni's VDN adapter path (same revision SGLD prefuses from).
    (VDN_REPO, VDN_REV, None),
]
