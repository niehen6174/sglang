# PR note: returned generation time

Base: `edee4308bcd7204d550234189d52a4cc25d93348`. Independent of the H3 integration, allocator and quantization fixes.

Populate `generation_time` for every member of each generated result group after the timing context exits and its duration becomes available. Previously the result could be populated while the timer still had its initial zero duration.

Three focused regression cases failed before and passed after the fix; 23 related tests passed. Subsequent full native API benchmark calls returned positive generation times. External API wall time remains independently measured and initialization is reported separately.

