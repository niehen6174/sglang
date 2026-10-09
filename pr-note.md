Base: `edee4308bcd7204d550234189d52a4cc25d93348`. Independent of the H3 ComfyUI feature branches.

# PR note: num_gpus must equal the dp*tp*sp*cfg layout

ServerArgs only checked divisibility, so `num_gpus=2` with tp=sp=dp=cfg=1 was accepted; rank 1 got no process group and blocked forever in `torch.distributed.new_group` (py-spy evidence saved; integrated and native both hung). tp=2 with sp=2 on 2 GPUs was also accepted and only failed late at runtime. Reject both for monolithic serving, and make `initialize_model_parallel` raise on every rank when `world_size != dit + vae` ranks. Tests fail before / pass after; exact layouts (tp2, sp2, dp2, auto) still accepted.
