Base: `edee4308bcd7204d550234189d52a4cc25d93348`. Independent of the H3 ComfyUI feature branches.

# PR note: reject dp_size > 1 in ComfyUI integrated mode

Integrated mode sends one request per sampler step with CUDA IPC tensors from the ComfyUI process. With DP2 on 2x H200, round-robin sent step 2 to the replica without the run's cached conditioning (`'list' object has no attribute 'ndim'`); pinning the run to one replica (tried, reverted) then failed on IPC inputs from cuda:0 reaching the replica on cuda:1. Behind ComfyUI's single prompt queue replicas never overlap, so DP cannot help; reject at configuration time and point to one ComfyUI instance per GPU or tp/sp. Native DiffGenerator DP2 is unaffected (verified bit-identical to single GPU).
