# ComfyUI SGLDiffusion Plugin

A ComfyUI plugin for SGLang Diffusion. Server mode talks to a standalone HTTP
server. Integrated mode keeps ComfyUI's CLIP / VAE / sampler loop and uses
SGLang only as a per-step DiT forward.

Integrated mode no longer ships dedicated `comfyui_*` pipelines. It starts the
native Flux / Qwen-Image / Z-Image / MiniMax-H3 / Wan pipeline under `--comfyui-mode`,
loads a single-file ComfyUI `.safetensors` (or GGUF overlay) through a
checkpoint spec, and translates each `apply_model` call through a small
per-model adapter.

## Installation

1. **Install SGLang**: Follow the [Installation Guide](https://docs.sglang.io/docs/sglang-diffusion/installation) to install `sglang[diffusion]`.
2. **Install Plugin**: Copy this entire directory (`ComfyUI_SGLDiffusion`) to your ComfyUI `custom_nodes/` folder.
3. **Restart ComfyUI**: Restart ComfyUI to load the plugin.

## Usage

The plugin supports two modes of operation: **Server Mode** (via HTTP API) and **Integrated Mode** (tight integration with ComfyUI).

### Supported Models
- **Z-Image**: High-speed image generation models (e.g., `Z-Image-Turbo`)
- **FLUX**: State-of-the-art text-to-image models (e.g., `FLUX.1-dev`)
- **Qwen-Image**: Multi-modal image generation models (e.g., `Qwen-Image`,`Qwen-Image-2512`). *Note: Image editing support is currently experimental and may have some issues.*
- **MiniMax-H3**: Joint video-and-audio DiT (`model_type=minimax_h3`). Integrated mode: T2V / I2VA / FL2VA use an `fl2va` checkpoint; R2V needs `ref2va`. CLIP and VAE stay in ComfyUI. Server mode uses `SGLDiffusion Generate MiniMax-H3`.

Wan integrated mode supports **Wan 2.1 T2V 1.3B** and **Wan 2.2 TI2V 5B**
(T2V and image-conditioned sampling). Use `model_type=auto-detect` or `wan2.1`
for both versions. Checkpoints use the original Wan tensor names, optionally
prefixed with `model.diffusion_model.`. ComfyUI owns UMT5 encoding, video latents,
frame masks, sampling and VAE decoding. The worker uses ComfyUI's selected DiT
precision unless `component_precisions` explicitly overrides it.

Wan batches, including CFG batches, currently run one row per worker request.
`enable_native_batch` does not enable packed Wan batching. Wan 2.1 I2V, VACE,
Animate, camera/control and serialized quantized checkpoints are outside this
integration's current scope.

### Mode 1: Server Mode (HTTP API)
Connect to a standalone SGLang Diffusion server.

1. **Start SGLang Diffusion Server**: Ensure the server is running and accessible.
2. **Connect to Server**: Use the `SGLDiffusion Server Model` node to connect (default: `http://localhost:3000/v1`).
3. **Generate Content**:
   - `SGLDiffusion Generate Image`: For text-to-image and image editing.
   - `SGLDiffusion Generate Video`: For text-to-video and image-to-video.
   - `SGLDiffusion Generate MiniMax-H3`: For joint video-and-audio on a standalone server.
4. **LoRA Support**: Use `SGLDiffusion Server Set LoRA` and `SGLDiffusion Server Unset LoRA`.

### Mode 2: Integrated Mode (Tight Integration)
ComfyUI keeps CLIP, VAE, and the sampler loop. SGLang loads only the DiT and
runs one forward per sampler step (pass-through scheduler, no text encode /
decode on the worker).

1. **Load Model**: Use the `SGLDiffusion UNET Loader` node to load your diffusion model.
2. **Configure Options**: Use the `SGLDiffusion Options` node to set runtime parameters like `num_gpus`, `tp_size`, `model_type`, or `enable_torch_compile`.
3. **Sample**: Connect the loaded model to standard ComfyUI samplers. Each step is packed by a model adapter and sent to the SGLang scheduler.
4. **LoRA Support**: Use the `SGLDiffusion LoRA Loader` for native LoRA integration.

## Adding a Model

Pick the mode before writing code; the wrong one costs several hundred lines
of weight mapping that buys nothing.

Take **Server Mode** when the model needs conditioning ComfyUI cannot supply
(audio, reference materials, task routing), emits more than one modality, or
has its own request contract. Reproducing that inside ComfyUI would duplicate
stages the server already runs.

- If the request fits the existing `generate_image` / `generate_video` fields,
  there is nothing to write — point the existing nodes at the server.
- If the model has extra request fields, pass them via `extra_fields`. The
  request schemas accept unknown keys, so `core/server_api.py` needs no
  per-model change.
- Add a node in `nodes.py` only to surface those inputs as ComfyUI widgets.
  `SGLDiffusionGenerateH3` is the worked example.

Take **Integrated Mode** only when the model denoises a single latent tensor
that ComfyUI already knows how to build and decode, so its KSampler can drive
the loop unchanged. Each model then needs:

- a checkpoint spec in `runtime/loader/comfyui_checkpoints/`, supplying
  architecture detection and the weight-name mapping for single-file checkpoints
- an adapter and executor in `executors/` translating the DiT forward to `Req`
- executor registration in `core/generator.py`, and native pipeline
  `pipeline_config_cls` / `sampling_params_cls` attributes for single-file loading

## Example Workflows

Reference workflow files are provided in the `workflows/` directory:

- **`flux_sgld_sp.json`**: Multi-GPU (Sequence Parallelism) workflow for FLUX models. High-performance inference across multiple cards.
- **`qwen_image_sgld.json`**: Qwen-Image generation with LoRA support. Optimized for multi-modal image tasks.
- **`z-image_sgld.json`**: High-speed image generation using Z-Image.
- **`wan21_t2v_sgld_api.json`**: Wan 2.1 1.3B text-to-video.
- **`wan22_t2v_sgld_api.json`**: Wan 2.2 5B text-to-video.
- **`wan22_i2v_sgld_api.json`**: Wan 2.2 5B image-to-video; replace the `EmptyImage` input with your image.

- **`sgld_text2img.json`**: Server-mode text-to-image generation with LoRA support.
- **`sgld_image2video.json`**: Server-mode image-to-video generation.
- **`minimax_h3_t2v_sgld.json`**: MiniMax-H3 T2V / I2VA / FL2VA (`fl2va` DiT).
- **`minimax_h3_r2v_sgld.json`**: MiniMax-H3 reference-to-video (`ref2va` DiT).
- **`minimax_h3_t2v_sgld_upscaler.json`**: H3 two-pass latent upscale (low-res then 3D ×2 refine).

The three Wan examples are **API-format graphs**, submitted as the `prompt`
field of `POST /prompt`. They use 20 UniPC steps, the simple scheduler, CFG 5,
flow shift 8,
512×320, 33 video frames and animated WebP output as a reduced-resolution
functional example, not a production quality preset. Install the matching DiT
in `diffusion_models`, UMT5 in `text_encoders`,
and Wan 2.1 / 2.2 VAE in `vae`.

Additional API examples preserve the published ComfyUI template defaults:

- `wan21_official_t2v_sgld_api.json`: 832×480, 33 frames, 30 UniPC steps, CFG 6, 16 FPS.
- `wan22_official_t2v_sgld_api.json`: 1280×704, 121 frames, 20 UniPC steps, CFG 5, 24 FPS.
- `wan22_official_i2v_sgld_api.json`: the same Wan 2.2 defaults with the template's image input enabled; set `LoadImage.image` to a file in ComfyUI's input directory (`example.png` is a placeholder).

These derive from Comfy-Org/workflow_templates revision
`8be1f8c4b5af2d550d70922a23b79cee599e1f3e`:
[text_to_video_wan.json](https://github.com/Comfy-Org/workflow_templates/blob/8be1f8c4b5af2d550d70922a23b79cee599e1f3e/templates/text_to_video_wan.json)
and [video_wan2_2_5B_ti2v.json](https://github.com/Comfy-Org/workflow_templates/blob/8be1f8c4b5af2d550d70922a23b79cee599e1f3e/templates/video_wan2_2_5B_ti2v.json).
They retain KSampler, prompts, published seeds, ModelSamplingSD3 shift 8,
UMT5, VAE decode, CreateVideo and SaveVideo. Only UNETLoader is replaced
and SGLDOptions added. Output filenames are changed for isolation; SaveVideo
format/codec widgets are translated to the installed ComfyUI 0.38.0 dynamic
API fields. All three paths passed both native and SGLD API execution on an
RTX 5090 32 GB without reducing resolution, frame count or sampling steps.
These are API graphs, not UI-format exports. Single-request end-to-end
SGLD times were higher than native; these checks establish compatibility,
not a speedup or pixel-identical output.

For other workflows supporting the models, you can easily use SGLang by replacing the official `UNET Loader` node with the `SGLDUNETLoader` node. Similarly, for LoRA support, replace the official LoRA loader with the `SGLDiffusion LoRA Loader`.

To use these workflows:
1. Open ComfyUI.
2. Load the workflow JSON file from the `workflows/` directory.
3. Adjust the parameters and model paths as needed.
4. Run the workflow.

## Current Implementation

SGLang's optimized kernels and parallelism (TP / SP, compile, cache) run on the
DiT only. Text encoding and VAE stay in ComfyUI.

## Architecture

```mermaid
flowchart LR
  subgraph comfy [ComfyUI process]
    CLIP[CLIP / text encode]
    VAE[VAE]
    SAMPLER[Sampler loop]
    EXEC["SGLDiffusionExecutor"]
    ADAPT["Model adapter<br/>pack / unpack"]
    CLIP --> SAMPLER
    SAMPLER --> EXEC --> ADAPT
  end

  ADAPT -->|CUDA IPC spill| R0

  subgraph sgl [SGLang]
    R0[Rank-0 scheduler]
    R0 -->|comfyui_mode and multi-rank| NCCL["Detach CUDA tensors<br/>NCCL broadcast"]
    R0 -->|otherwise| PYO["Original SP / CFG / TP<br/>broadcast_pyobj"]
    NCCL --> PIPE
    PYO --> PIPE
    PIPE["Native pipeline<br/>--comfyui-mode"]
    SPEC["Checkpoint spec<br/>single .safetensors"] --> PIPE
    PIPE --> STAGE["Latent prep + session cache<br/>+ DenoisingStage"]
  end

  STAGE -->|noise_pred IPC| EXEC
  STAGE --> VAE
```

Per sampler step:

1. The adapter turns ComfyUI `apply_model` tensors into an SGLang `Req`.
2. Local ZMQ pickle replaces CUDA tensors with IPC handles so latents stay on GPU.
3. Rank 0 materializes the handles. Multi-rank `--comfyui-mode` then detaches CUDA tensors and broadcasts them over NCCL; the general SP / CFG / TP path is still the original whole-list `broadcast_pyobj`.
4. The worker pipeline is the native model class with modules trimmed to `transformer` + pass-through scheduler. A single-file checkpoint goes through `comfyui_checkpoints`. After the first step, conditioning stays in a worker session; later steps send latents and the timestep.

Adding a model means a checkpoint spec plus a `ComfyUIModelAdapter`. There is no extra ComfyUI pipeline class.
