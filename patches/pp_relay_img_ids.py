#!/usr/bin/env python3
"""Graft the backport's dsv4_img_ids PP relay into the dsv41reap overlay
model.py: vision-capable checkpoints create gate.bias_vl on every MoE layer,
and the FFN raises without input_ids on downstream PP ranks."""
import sys
path = sys.argv[1]; src = open(path).read()
if "dsv4_img_ids" in src:
    print("already patched"); sys.exit(0)

# 1) dummy/empty intermediate tensors: declare the relay
old1 = '''            "pre_mix": torch.zeros(
                (batch_size, self.hc_mult),
                dtype=torch.float32,
                device=device,
            ),
        }'''
new1 = '''            "pre_mix": torch.zeros(
                (batch_size, self.hc_mult),
                dtype=torch.float32,
                device=device,
            ),
            # PP relay for vision-MoE routing (bias_vl consumes input_ids
            # on every MoE layer, downstream ranks included).
            "dsv4_img_ids": torch.zeros(
                (batch_size,), dtype=torch.int64, device=device
            ),
        }'''
assert src.count(old1) == 1, ("edit1", src.count(old1))
src = src.replace(old1, new1)

# 2) forward: restore input_ids on downstream ranks
old2 = '''        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:'''
new2 = '''        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            if input_ids is None:
                input_ids = intermediate_tensors["dsv4_img_ids"]

        if self.use_mega_moe:'''
assert src.count(old2) == 1, ("edit2", src.count(old2))
src = src.replace(old2, new2)

# 3) ship site: forward input_ids to the next rank
old3 = '''            if plan.ship_cand:
                assert self.candidate_block_buffer is not None
                out[_DSV41_CAND_KEY] = self.candidate_block_buffer[:num_tokens]
            return IntermediateTensors(out)'''
new3 = '''            if plan.ship_cand:
                assert self.candidate_block_buffer is not None
                out[_DSV41_CAND_KEY] = self.candidate_block_buffer[:num_tokens]
            out["dsv4_img_ids"] = (
                input_ids.to(torch.int64)
                if input_ids is not None
                else torch.zeros(
                    num_tokens, dtype=torch.int64, device=positions.device
                )
            )
            return IntermediateTensors(out)'''
assert src.count(old3) == 1, ("edit3", src.count(old3))
src = src.replace(old3, new3)

open(path, "w").write(src); print("patched", path)
