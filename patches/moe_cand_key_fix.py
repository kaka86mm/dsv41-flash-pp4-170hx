#!/usr/bin/env python3
"""PP: vllm v1/worker/gpu/model_runner.py builds a non-first rank's
IntermediateTensors by copying every key of its OWN dummy spec from the
received dict — but keys the rank *produces* (dsv41_cand when ship_cand)
never arrive from the previous rank. Skip them; forward() overwrites via
its ship buffers anyway.
usage: moe_cand_key_fix.py <vllm/v1/worker/gpu/model_runner.py>"""
import sys
path = sys.argv[1]; src = open(path).read()
if "dsv41: keys this rank *produces*" in src:
    print("already patched"); sys.exit(0)
old = '''            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
            }'''
new = '''            # dsv41: keys this rank *produces* (e.g. dsv41_cand when
            # ship_cand) are declared in its own spec but never arrive from
            # the previous rank — skip them; forward overwrites via its ship
            # buffers anyway.
            new_tensors = {
                k: v[:n]
                if dummy_run
                else v[:n].copy_(intermediate_tensors.tensors[k][:n])
                for k, v in self.intermediate_tensors.tensors.items()
                if dummy_run or k in intermediate_tensors.tensors
            }'''
assert src.count(old) == 1, src.count(old)
open(path, "w").write(src.replace(old, new)); print("patched", path)
