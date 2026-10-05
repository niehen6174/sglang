"""
Base executor class for SGLang Diffusion ComfyUI integration.
"""

import hashlib
import uuid

import torch

try:
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
except ImportError:
    print(
        "Error: sglang.multimodal_gen is not installed. Please install it using 'pip install sglang[diffusion]'"
    )


def _hash_value(digest, value) -> None:
    """Hash ``value`` into ``digest``, framing each item so values cannot merge."""
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        digest.update(f"T{tensor.dtype}{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(f"D{len(value)}".encode())
        for key, item in value.items():
            _hash_value(digest, key)
            _hash_value(digest, item)
    elif isinstance(value, (list, tuple)):
        digest.update(f"L{len(value)}".encode())
        for item in value:
            _hash_value(digest, item)
    else:
        text = repr(value).encode()
        digest.update(f"V{len(text)}:".encode())
        digest.update(text)


def _row(value, index: int, batch: int):
    if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch:
        return value[index : index + 1]
    if type(value) in (list, tuple):
        return type(value)(_row(item, index, batch) for item in value)
    return value


def _uniform(value):
    if not torch.is_tensor(value) or value.numel() <= 1:
        return True
    return bool((value == value.reshape(-1)[0]).all().item())


def _reject_model_patches(transformer_options) -> None:
    # The DiT runs in the SGLD worker, so ComfyUI block/attention patches would
    # be dropped silently (e.g. MiniMax H3 Fun ControlNet, ModelAttentionBackend).
    opts = transformer_options or {}
    found = [
        f"patches_replace.{name}"
        for name, blocks in (opts.get("patches_replace") or {}).items()
        if blocks
    ]
    found += [
        f"patches.{name}"
        for name, items in (opts.get("patches") or {}).items()
        if items
    ]
    if opts.get("optimized_attention_override") is not None:
        found.append("optimized_attention_override")
    if found:
        raise ValueError(
            "SGLD integrated mode runs the diffusion model in its worker and cannot "
            f"apply ComfyUI model patches ({', '.join(sorted(found))}); remove the "
            "patch nodes (ControlNet, attention backend, block patches) or use the "
            "native ComfyUI loader"
        )


class SGLDiffusionExecutor(torch.nn.Module):
    """Shared ComfyUI DiT-forward executor. Per-model logic lives on the adapter."""

    adapter_cls = None

    def __init__(self, generator, model_path, model, config):
        super(SGLDiffusionExecutor, self).__init__()
        self.generator = generator
        self.model_path = model_path
        self.model = model
        self.dtype = config.unet_config["dtype"]
        self.config = config
        self.loras = []
        self._lora_input = None
        self._sgld_reload = None
        self._ensure_runtime = None
        if self.adapter_cls is None:
            raise TypeError(f"{type(self).__name__} must set adapter_cls")
        self.adapter = self.adapter_cls()
        self.session_id = uuid.uuid4().hex
        self._run_id = 0
        self._sent_conds: set[str] = set()

    @staticmethod
    def should_suppress_logs(timestep):
        """Determine if logs should be suppressed based on timestep value."""
        if torch.is_tensor(timestep):
            return bool((timestep < 1.0).all().item())
        return bool(timestep < 1.0)

    def set_lora(self, lora_nickname=None, lora_path=None, strength=None, target=None):
        """Set LoRA adapter using SGLang Diffusion API."""
        desired = {
            "lora_nickname": lora_nickname,
            "lora_path": lora_path,
            "strength": strength,
            "target": target,
        }
        if lora_nickname and len(lora_nickname) > 0:
            self.generator.set_lora(
                lora_nickname=lora_nickname,
                lora_path=lora_path,
                strength=strength,
                target=target,
            )
        self._lora_input = desired

        self._lora_input = desired

    def begin_sampler_run(self) -> None:
        """One ComfyUI ``sampler.sample()`` invocation is one cache lifetime."""
        self._run_id += 1
        self._sent_conds = set()

    def end_sampler_run(self) -> None:
        """Run cache is evicted on the next bind of a newer id for this executor."""

    def sampler_sample_wrapper(self, executor, *args, **kwargs):
        ensure = getattr(self, "_ensure_runtime", None)
        if ensure is not None:
            ensure(self)
        model_wrap = args[0] if args else kwargs.get("model_wrap")
        patcher = getattr(model_wrap, "model_patcher", None)
        if patcher is not None:
            flags = patcher.model_options.get("sgld_request_flags", {})
            self.enable_cache_dit = flags.get("enable_cache_dit")
            self.cache_dit_params = flags.get("cache_dit_params")
            self.request_options = dict(flags.get("request_options", {}))
            desired = patcher.model_options.get("sgld_lora_input")
            if desired != self._lora_input:
                if self._lora_input is not None:
                    self.generator.unmerge_lora_weights()
                    self._lora_input = None
                if desired is not None:
                    self.set_lora(**desired)
        self.begin_sampler_run()
        try:
            return executor(*args, **kwargs)
        finally:
            self.end_sampler_run()

    def comfyui_session_id(self) -> str:
        return f"{self.session_id}:{self._run_id}"

    def _cond_key(self, packed) -> str | None:
        embeds = packed.prompt_embeds
        if not embeds:
            return None
        tensor = embeds[0]
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return None
        # Hash everything drop_cached_fields removes: a hit means the worker
        # restores all of it, so a partial key would revive another cond.
        digest = hashlib.blake2b(digest_size=16)
        _hash_value(
            digest,
            (
                embeds,
                packed.pooled_embeds,
                packed.prompt_seq_lens,
                {
                    key: packed.extra_req.get(key)
                    for key in self.adapter.cached_extra_keys
                },
            ),
        )
        return digest.hexdigest()

    def _mark_and_maybe_drop(self, packed) -> None:
        key = self._cond_key(packed)
        if key is not None:
            packed.extra_req["comfyui_cond_key"] = key
            if key in self._sent_conds:
                self.adapter.drop_cached_fields(packed)
            else:
                self._sent_conds.add(key)

    def _sampling_params_kwargs(self, packed, timestep) -> dict:
        return {
            "prompt": " ",
            "num_outputs_per_prompt": (
                int(packed.latents.shape[0])
                if self.adapter.supports_batched_forward
                else 1
            ),
            "guidance_scale": packed.guidance_scale,
            "height": packed.height,
            "width": packed.width,
            "num_frames": 1,
            "num_inference_steps": 1,
            "save_output": False,
            "suppress_logs": self.should_suppress_logs(timestep),
        }

    def _execute_packed(self, packed, x, timestep):
        ensure = getattr(self, "_ensure_runtime", None)
        if ensure is not None:
            ensure(self)
        self._mark_and_maybe_drop(packed)
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            **self._sampling_params_kwargs(packed, timestep),
        )
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        self.adapter.fill_req(req, packed)
        extra = dict(req.extra or {})
        extra["comfyui_session_id"] = self.comfyui_session_id()
        for key in ("comfyui_cond_key", "comfyui_cache_fp"):
            value = packed.extra_req.get(key)
            if value is not None:
                extra[key] = value
        req.extra = extra
        req.generator = [torch.Generator("cuda") for _ in range(req.batch_size)]
        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        if output_batch.error:
            raise RuntimeError(f"SGLang Diffusion worker failed: {output_batch.error}")
        return self.adapter.unpack(output_batch.noise_pred, packed, x)

    def forward(self, x, timestep, context, **kwargs):
        _reject_model_patches(kwargs.get("transformer_options"))
        batch = int(x.shape[0]) if torch.is_tensor(x) else 1
        if batch > 1:
            if (
                self.adapter.supports_batched_forward
                and self.generator.server_args.comfyui_native_batch
                and _uniform(timestep)
                and _uniform(kwargs.get("guidance"))
            ):
                packed = self.adapter.pack(x, timestep, context, **kwargs)
                # The worker expects a schedule, not one timestep per sample.
                packed.timesteps = packed.timesteps.reshape(-1)[:1]
                return self._execute_packed(packed, x, timestep)
            return torch.cat(
                [
                    self._forward_one(
                        _row(x, i, batch),
                        _row(timestep, i, batch),
                        _row(context, i, batch),
                        **{key: _row(value, i, batch) for key, value in kwargs.items()},
                    )
                    for i in range(batch)
                ]
            )
        return self._forward_one(x, timestep, context, **kwargs)

    def _forward_one(self, x, timestep, context, **kwargs):
        packed = self.adapter.pack(x, timestep, context, **kwargs)
        return self._execute_packed(packed, x, timestep)
