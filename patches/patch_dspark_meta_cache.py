#!/usr/bin/env python3
"""Cache the per-step python churn of DSparkProposer.set_inputs_first_pass.

Measured (DSV41_DEBUG_DSPARK=1, thinking decode): inputs+meta runs 5.6-6.1
ms/call, ~35% of the whole draft cost, and most of it is re-creating tensors
that are identical every step at a fixed decode shape:

  - scratch_indices     torch.empty every call (content rewritten by the
                        copy_and_expand kernel anyway)
  - token_indices       arange[:n].to(int64) — a fresh alloc + cast per step;
                        a full-length int64 arange built once slices to views
  - qsl_cpu             numpy -> tensor clone + multiply per step; depends only
                        on batch_size
  - seq_lens + nq       new GPU tensor per step; a persistent buffer updated
                        in place is what the rest of the pipeline already does
                        (self.input_ids et al)

Decode-only caches, keyed by batch_size, bounded by max_num_seqs. The attention
metadata build() itself is left untouched (its cost is sequence-state
dependent). PIECEWISE graphs: python between graph pieces runs on every step,
so caching python objects is replay-safe; all cached tensors follow the
existing persistent-buffer-refresh pattern.

usage: patch_dspark_meta_cache.py <hybrid/dspark_proposer.py>  (NOT applied by
default; run it in a maintenance window and measure with DSV41_DEBUG_DSPARK=1)
"""
import sys
path = sys.argv[1]; src = open(path).read()
if "dsv41: draft meta cache" in src:
    print("already patched"); sys.exit(0)

# 1) cache state next to the existing buffers in __init__ (anchor on the
#    confidence env read; anything single-line and stable works)
old1 = '        self._conf_default = float(os.environ.get("DSV41_DSPARK_CONF", "0") or 0)\n'
new1 = old1 + '''        # dsv41: draft meta cache — per-step decode buffers keyed by batch_size
        self._dsv41_meta_cache = {}
        self._dsv41_scratch_indices = None
        self._dsv41_arange_i64 = None
        self._dsv41_seq_lens_buf = None
'''
assert src.count(old1) == 1, ("init anchor", src.count(old1))
src = src.replace(old1, new1, 1)

# 2) the four per-step allocations -> cached/persistent variants
old2 = '''        scratch_indices = torch.empty(
            max(1, batch_size * (nq - 1)), dtype=torch.int32, device=self.device
        )'''
new2 = '''        # dsv41: draft meta cache — scratch content is fully rewritten by the
        # kernel below; a grow-on-demand buffer replaces the per-step alloc.
        if (
            self._dsv41_scratch_indices is None
            or self._dsv41_scratch_indices.numel() < max(1, batch_size * (nq - 1))
        ):
            self._dsv41_scratch_indices = torch.empty(
                max(1, batch_size * (nq - 1)), dtype=torch.int32, device=self.device
            )
        scratch_indices = self._dsv41_scratch_indices'''
assert src.count(old2) == 1, ("scratch anchor", src.count(old2))
src = src.replace(old2, new2, 1)

old2b = '''        else:
            token_indices_to_sample = scratch_indices.to(torch.int64)'''
new2b = '''        else:
            # dsv41: draft meta cache — the persistent scratch may be longer
            # than this step needs (grown for a larger batch earlier); convert
            # exactly this step's span so stale tail lanes never leak in.
            token_indices_to_sample = scratch_indices[: max(1, batch_size * (nq - 1))].to(torch.int64)'''
assert src.count(old2b) == 1, ("scratch else anchor", src.count(old2b))
src = src.replace(old2b, new2b, 1)

old3 = '''        if self.sample_from_anchor:
            token_indices_to_sample = self.arange[:num_query_total].to(torch.int64)'''
new3 = '''        if self.sample_from_anchor:
            # dsv41: draft meta cache — one int64 arange, sliced per step (views).
            if self._dsv41_arange_i64 is None or self._dsv41_arange_i64.numel() < self.arange.numel():
                self._dsv41_arange_i64 = self.arange.to(torch.int64)
            token_indices_to_sample = self._dsv41_arange_i64[:num_query_total]'''
assert src.count(old3) == 1, ("arange anchor", src.count(old3))
src = src.replace(old3, new3, 1)

old4 = '''        new_query_start_loc = self.arange[: batch_size + 1] * nq
        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu'''
new4 = '''        new_query_start_loc = self.arange[: batch_size + 1] * nq
        # dsv41: draft meta cache — seq_lens into a persistent buffer, updated
        # in place (same refresh pattern as input_ids/positions above).
        if self._dsv41_seq_lens_buf is None or self._dsv41_seq_lens_buf.shape[0] < cad.seq_lens.shape[0]:
            self._dsv41_seq_lens_buf = torch.empty_like(cad.seq_lens)
        effective_seq_lens = self._dsv41_seq_lens_buf[: cad.seq_lens.shape[0]]
        effective_seq_lens.copy_(cad.seq_lens)
        if has_num_rejected:
            effective_seq_lens.sub_(num_rejected_tokens_gpu)'''
assert src.count(old4) == 1, ("seq_lens anchor", src.count(old4))
src = src.replace(old4, new4, 1)

# 3) downstream: 'effective_seq_lens + nq' now aliases the buffer; make the
#    add non-mutating so the cached buffer survives to the next step intact
#    only if someone keeps a reference — the cad stores the sum, so give it
#    its own persistent buffer refreshed here as well (simplest correct form:
#    keep a second buffer for the +nq result).
old5 = '''        upper = (
            cad.seq_lens_cpu_upper_bound + nq
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        qsl_cpu = torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * nq'''
new5 = '''        effective_seq_lens.add_(nq)
        upper = (
            cad.seq_lens_cpu_upper_bound + nq
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        # dsv41: draft meta cache — qsl_cpu depends only on batch_size.
        qsl_cpu = self._dsv41_meta_cache.get(("qsl", batch_size))
        if qsl_cpu is None:
            qsl_cpu = torch.from_numpy(
                self.token_arange_np[: batch_size + 1]
            ).clone() * nq
            self._dsv41_meta_cache[("qsl", batch_size)] = qsl_cpu'''
assert src.count(old5) == 1, ("qsl anchor", src.count(old5))
src = src.replace(old5, new5, 1)

old6 = '''        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens + nq,'''
new6 = '''        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=effective_seq_lens,'''
assert src.count(old6) == 1, ("cad anchor", src.count(old6))
src = src.replace(old6, new6, 1)

open(path, "w").write(src); print("patched", path)
