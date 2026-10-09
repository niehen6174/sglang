"""
Base executor class for SGLang Diffusion ComfyUI integration.
"""

import os
import uuid

import torch

try:
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
        COMFYUI_CFG_SPLIT_KEY,
    )
except ImportError:
    print(
        "Error: sglang.multimodal_gen is not installed. Please install it using 'pip install sglang[diffusion]'"
    )


# Arbitrary; allocator slack and transient copies while the worker loads LoRA.
_LORA_VRAM_HEADROOM = 2 * 1024**3


def release_comfy_vram(nbytes: int) -> None:
    """Evict ComfyUI-held models until ``nbytes`` of device memory are free.

    The SGLD worker allocates outside ComfyUI's accounting, so ComfyUI does
    not evict, e.g., a text encoder loaded earlier in the graph on its behalf.
    Evicted models are reloaded on demand.
    """
    try:
        from comfy import model_management
    except ImportError:
        return
    device = model_management.get_torch_device()
    model_management.free_memory(nbytes, device)
    model_management.soft_empty_cache()


def evict_comfy_models(keep=None) -> None:
    """Unload every ComfyUI-held model on the device except ``keep``.

    ``free_memory(n)`` only partially unloads until ComfyUI sees ``n`` bytes
    free, and the worker's activations are invisible to that accounting, so a
    text encoder left resident earlier in the graph starves the worker.
    Evicted models reload on demand.
    """
    try:
        from comfy import model_management
    except ImportError:
        return
    keep_loaded = [
        loaded
        for loaded in model_management.current_loaded_models
        if keep is not None and loaded.model is keep
    ]
    model_management.free_memory(
        1e30, model_management.get_torch_device(), keep_loaded=keep_loaded
    )
    model_management.soft_empty_cache()


def _lora_bytes(lora_path) -> int:
    paths = lora_path if isinstance(lora_path, (list, tuple)) else [lora_path]
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except (OSError, TypeError):
            pass
    return total


class SGLDiffusionExecutor(torch.nn.Module):
    """Shared ComfyUI DiT-forward executor. Per-model logic lives on the adapter."""

    adapter_cls = None
    # LoRA merge mode for set_lora; None keeps the server default.
    lora_merge_mode: str | None = None
    # Whether the worker pipeline routes CFG-split sub-calls (see cfg_split.py).
    supports_cfg_split: bool = False

    def __init__(self, generator, model_path, model, config):
        super(SGLDiffusionExecutor, self).__init__()
        self.generator = generator
        self.model_path = model_path
        # Not a registered submodule: the ComfyUI model owns this executor as
        # its diffusion_model, and a module cycle breaks state_dict() / _apply().
        object.__setattr__(self, "model", model)
        # DiT parameter names from the checkpoint header, so ComfyUI's LoRA key
        # mapping sees the served model (see state_dict below).
        self.dit_state_keys = tuple(getattr(model, "sgld_dit_state_keys", ()))
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
        # Cond keys whose conditioning every worker rank has received.
        self._sent_conds: set[tuple] = set()
        # CFG split: cond key -> CFG ranks that received it in a split step.
        self._sent_cond_slots: dict[tuple, set[int]] = {}
        # CFG ranks that each run one cond of a step; 1 disables the split.
        self.cfg_split_ranks = 1
        self._cfg_split_mode: str | None = None
        self._cfg_split_records: list = []
        self._cfg_split_outputs: list = []

    @classmethod
    def validate_sgld_options(cls, sgld_options: dict | None) -> None:
        """Reject SGLDOptions this model cannot honour, before the worker starts."""
        cls.cfg_split_ranks_for(sgld_options)

    @classmethod
    def cfg_split_ranks_for(cls, sgld_options: dict | None) -> int:
        """CFG ranks that split a step's conds: 2 with enable_cfg_parallel, else 1.

        Executors without ``supports_cfg_split`` keep the plain worker
        behaviour for enable_cfg_parallel (every CFG rank runs every call).
        """
        options = sgld_options or {}
        degree = options.get("cfg_parallel_degree") or 1
        if not cls.supports_cfg_split or not (
            options.get("enable_cfg_parallel") or degree > 1
        ):
            return 1
        for name in ("sp_degree", "ulysses_degree", "ring_degree", "tp_size"):
            value = options.get(name)
            if value is not None and value not in (-1, 1):
                raise ValueError(
                    f"CFG split (enable_cfg_parallel) runs one cond per GPU and "
                    f"cannot be combined with {name}={value}; use one of them"
                )
        num_gpus = options.get("num_gpus") or 1
        if num_gpus != 2 or degree not in (1, 2):
            raise ValueError(
                "CFG split (enable_cfg_parallel) runs ComfyUI's cond and uncond on "
                f"one GPU each and needs num_gpus=2 (got num_gpus={num_gpus}"
                + (f", cfg_parallel_degree={degree}" if degree > 1 else "")
                + ")"
            )
        return 2

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Header-only view of the served DiT: names with empty placeholders.

        The weights live in the SGLD worker. ComfyUI builds LoRA key maps from
        ``model.state_dict()``; exposing the names lets SGLDModelPatcher reject
        a native LoRA that targets this DiT instead of dropping it silently.
        """
        if destination is None:
            destination = {}
        for key in self.dit_state_keys:
            destination[prefix + key] = torch.empty(0)
        return destination

    @staticmethod
    def should_suppress_logs(timestep):
        """Determine if logs should be suppressed based on timestep value."""
        if torch.is_tensor(timestep):
            # ComfyUI batches cond/uncond rows, so the timestep can be [B].
            return bool((timestep.reshape(-1)[0] < 1.0).item())
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
            # Adapter tensors are placed on the worker's device.
            release_comfy_vram(2 * _lora_bytes(lora_path) + _LORA_VRAM_HEADROOM)
            self.generator.set_lora(
                lora_nickname=lora_nickname,
                lora_path=lora_path,
                strength=strength,
                target=target,
                merge_mode=self.lora_merge_mode,
            )

        self._lora_input = desired

    def begin_sampler_run(self) -> None:
        """One ComfyUI ``sampler.sample()`` invocation is one cache lifetime."""
        self._run_id += 1
        self._sent_conds = set()
        self._sent_cond_slots = {}

    def end_sampler_run(self) -> None:
        """Run cache is evicted on the next bind of a newer id for this executor."""

    def sampler_sample_wrapper(self, executor, *args, **kwargs):
        if self._ensure_runtime is not None:
            self._ensure_runtime(self)
        model_wrap = args[0] if args else kwargs["model_wrap"]
        # The worker samples next; ComfyUI models reload when the graph needs them.
        evict_comfy_models(keep=model_wrap.model_patcher)
        desired = model_wrap.model_patcher.model_options.get("sgld_lora_input")
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

    def _cond_key(self, packed) -> tuple | None:
        embeds = packed.prompt_embeds
        if not embeds:
            return None
        tensor = embeds[0]
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            return None
        flat = tensor.reshape(-1)
        return (
            tuple(int(dim) for dim in tensor.shape),
            float(flat[0].item()),
            float(flat[-1].item()),
        )

    def _mark_and_maybe_drop(self, packed, slot: int | None = None) -> None:
        """Send a cond's conditioning once per run to every rank that will need it.

        ``slot`` is the CFG rank that alone runs this call in a split step;
        None means every rank runs it.
        """
        key = self._cond_key(packed)
        if key is None:
            return
        packed.extra_req["comfyui_cond_key"] = repr(key)
        slots = self._sent_cond_slots.setdefault(key, set())
        if key in self._sent_conds or (slot is not None and slot in slots):
            self.adapter.drop_cached_fields(packed)
        elif slot is None:
            self._sent_conds.add(key)
        else:
            slots.add(slot)

    def _sampling_params_kwargs(self, packed, timestep) -> dict:
        return {
            "prompt": " ",
            "guidance_scale": packed.guidance_scale,
            "height": packed.height,
            "width": packed.width,
            "num_frames": 1,
            "num_inference_steps": 1,
            "save_output": False,
            "suppress_logs": self.should_suppress_logs(timestep),
        }

    def _fill_req(self, req, packed) -> None:
        self.adapter.fill_req(req, packed)
        extra = dict(req.extra or {})
        extra["comfyui_session_id"] = self.comfyui_session_id()
        for key in ("comfyui_cond_key", "comfyui_cache_fp"):
            value = packed.extra_req.get(key)
            if value is not None:
                extra[key] = value
        req.extra = extra

    def _new_req(self, packed, timestep):
        ensure = getattr(self, "_ensure_runtime", None)
        if ensure is not None:
            ensure(self)
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.model_path,
            server_args=self.generator.server_args,
            **self._sampling_params_kwargs(packed, timestep),
        )
        req = prepare_request(
            server_args=self.generator.server_args,
            sampling_params=sampling_params,
        )
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]
        return req

    def _send(self, req):
        output_batch = self.generator._send_to_scheduler_and_wait_for_response([req])
        if output_batch.noise_pred is None:
            # The worker reports a failed step as an error string, not an exception.
            raise RuntimeError(
                "SGLD worker failed this DiT step: "
                f"{output_batch.error or 'no noise_pred returned'}"
            )
        return output_batch.noise_pred

    def _execute_packed(self, packed, x, timestep):
        req = self._new_req(packed, timestep)
        self._mark_and_maybe_drop(packed)
        self._fill_req(req, packed)
        return self.adapter.unpack(self._send(req), packed, x)

    def execute_cfg_split(self, records) -> list:
        """Run the recorded calls of one step together, call ``i`` on CFG rank ``i``.

        The request carries every call's ``Req`` fields under
        COMFYUI_CFG_SPLIT_KEY; the worker answers with one velocity per call.
        """
        if len(records) != self.cfg_split_ranks:
            raise ValueError(
                f"CFG split needs {self.cfg_split_ranks} calls, got {len(records)}"
            )
        req = self._new_req(records[0][0], records[0][2])
        calls = []
        for slot, (packed, _, _) in enumerate(records):
            self._mark_and_maybe_drop(packed, slot=slot)
            fields = _ReqFields()
            self._fill_req(fields, packed)
            calls.append(fields.values)
        self._fill_req(req, records[0][0])
        req.extra = {**req.extra, COMFYUI_CFG_SPLIT_KEY: calls}
        outputs = self._send(req)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(records):
            raise RuntimeError(
                "SGLD worker did not run the CFG split request (one velocity per "
                "call expected); is the worker started with enable_cfg_parallel?"
            )
        return [
            self.adapter.unpack(out, packed, x)
            for out, (packed, x, _) in zip(outputs, records)
        ]

    def calc_cond_batch_wrapper(
        self, executor, model, conds, x_in, timestep, model_options
    ):
        """ComfyUI CALC_COND_BATCH wrapper: cond and uncond on separate CFG ranks."""
        from .cfg_split import calc_cond_batch_cfg_split

        if self.cfg_split_ranks > 1 and model.diffusion_model is self:
            out = calc_cond_batch_cfg_split(
                self, model, conds, x_in, timestep, model_options
            )
            if out is not None:
                return out
        return executor(model, conds, x_in, timestep, model_options)

    def forward(self, x, timestep, context, **kwargs):
        if self._cfg_split_mode == "replay":
            return self._replay_cfg_split(x)
        packed = self.adapter.pack(x, timestep, context, **kwargs)
        if self._cfg_split_mode == "record":
            self._cfg_split_records.append((packed, x, timestep))
            return self.adapter.placeholder_output(x)
        return self._execute_packed(packed, x, timestep)

    def _replay_cfg_split(self, x):
        out = self._cfg_split_outputs.pop(0)
        shapes = [
            tuple(t.shape) for t in (out if isinstance(out, (list, tuple)) else [out])
        ]
        expect = [tuple(t.shape) for t in (x if isinstance(x, (list, tuple)) else [x])]
        if shapes != expect:
            raise RuntimeError(
                f"CFG split replay out of order: velocity {shapes} for input {expect}"
            )
        return out


class _ReqFields:
    """Records the ``Req`` attributes an adapter's fill_req sets."""

    def __init__(self):
        object.__setattr__(self, "values", {"extra": {}})

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self.values[name] = value
