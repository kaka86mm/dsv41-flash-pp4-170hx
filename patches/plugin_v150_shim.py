#!/usr/bin/env python3
"""Adapt vllm-exl3 to exllamav3 v1.5.0's exl3_moe: the old entry gained five
positional tail params (output_scratch, fused_base, count_lo, count_hi,
m_tile) for the tiered per-expert dispatch. Mirror the reference defaults
from block_sparse_mlp.run_fused."""
import sys
path = sys.argv[1]; src = open(path).read()
if "_exl3_moe_v15_tail" in src:
    print("already patched"); sys.exit(0)

old1 = '''def pin_exl3_expert_map('''
new1 = '''def _exl3_moe_v15_tail(fn, fused_rows: int):
    """exllamav3 v1.5.0 appends output_scratch/fused_base/count_lo/count_hi/
    m_tile to exl3_moe (tiered per-expert dispatch); older builds stop at
    num_active. Reference defaults: block_sparse_mlp.run_fused."""
    # pybind docs are numbered args (arg0, arg1...), not names — probe the
    # v1.5.0 marker binding instead.
    try:
        import exllamav3_ext
        if not hasattr(exllamav3_ext, "exl3_moe_coop"):
            return None  # pre-1.5.0: old entry without the tail
    except Exception:
        return None
    return (None, None, 1, int(fused_rows), 16)


def pin_exl3_expert_map('''
assert src.count(old1) == 1, ("helper anchor", src.count(old1))
src = src.replace(old1, new1)

old2 = '''    if n_active_host is not None:
        fn(*args, n_active_host)
    else:
        fn(*args)
'''
new2 = '''    v15_tail = _exl3_moe_v15_tail(fn, TEMP_ROWS_FUSED)
    if v15_tail is not None:
        # v1.5.0: num_active positional, then the tiered-dispatch tail.
        fn(*args, -1 if n_active_host is None else n_active_host, *v15_tail)
    elif n_active_host is not None:
        fn(*args, n_active_host)
    else:
        fn(*args)
'''
assert src.count(old2) == 1, ("call anchor", src.count(old2))
src = src.replace(old2, new2)

open(path, "w").write(src); print("patched", path)
