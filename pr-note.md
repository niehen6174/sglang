# PR note: H3 online MXFP8 loading

Base: `edee4308bcd7204d550234189d52a4cc25d93348`. Independent loader fix; not dependent on ComfyUI feature changes.

For CPU-backed, non-FSDP H3 online MXFP8 loading, stage each linear module on CUDA for quantization postprocessing and restore CPU placement. Avoid moving the complete BF16 model to the GPU first. Gate the path to the supported online case; preserve serialized quantization, FSDP, and other model loading behavior.

Validation: 13 new CPU checks and one actual GPU check passed; 26 existing CPU loading regressions passed. Final ComfyUI and native full-pipeline API groups each completed seven requests (five measured hot requests) in the combined workspace. On RTX 5090, `fp8` maps to MXFP8; this is not a generic FP8 quality claim. VDN MXFP8 was faster but changed video/audio latent trajectories substantially and needs content-specific quality assessment.

