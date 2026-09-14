#!/usr/bin/env python3
"""Port v19 backport's sm80 engram fix into overlay engram.py: the Triton
lookup kernel loads the e4m3 table through a uint8 view and decodes with
fp8_sm80 ALU helpers — tl fp8e4nv loads don't compile below SM89."""
import sys
path = sys.argv[1]; src = open(path).read()
if "_decode_fp8_f32" in src:
    print("already patched"); sys.exit(0)

# 1) import the sm80 decode helper
old = "from vllm.logger import init_logger\n"
new = ("from vllm.logger import init_logger\n"
       "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32\n")
assert src.count(old) == 1; src = src.replace(old, new)

# 2) kernel: load uint8 and decode (weight is passed as a uint8 view)
old = '''        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )'''
new = '''        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0,
        )
        values = _decode_fp8_f32(values, False)'''
assert src.count(old) == 1, src.count(old); src = src.replace(old, new)

# 3) kernel: decoded values are already f32
old = "(values.to(tl.float32) * scale).to(tl.bfloat16),"
new = "(values * scale).to(tl.bfloat16),"
assert src.count(old) == 1; src = src.replace(old, new)

# 4) launch site: hand the kernel the uint8 view of the e4m3 table
old = '''        _engram_lookup_kernel[(grid,)](
            weight,'''
new = '''        _engram_lookup_kernel[(grid,)](
            weight.view(torch.uint8),'''
assert src.count(old) == 1; src = src.replace(old, new)

open(path, "w").write(src); print("patched", path)
