"""
CUDA-graph execution of the WEFT pipeline (engine weft_graph in engines.py).

Eager WEFT launches ~60 kernels per query from Python and synchronizes twice inside sparse
Stage 3 (to size the sort, then to size the reduce output), so on LoTTE the GPU is busy for only
~0.44 of the ~0.65 ms wall time. Here a query is two CUDA graph replays and one sync:

  phase A (graph per nc): query prep (native fp16 centroid scores, imputation, compact scores)
                          + Stage 1 + Stage 2. Writes the posting count P on the device.
  host:                   P = total_postings.item()            <- the only mid-query sync
  phase B (graph per (cap, ndocs)): sync-free sparse Stage 3 over a fixed capacity cap >= P + 1
                          (the P keys, then sentinels), top-ndocs, TileMaxSim rerank, filter,
                          top-k. cap is P rounded up to 1/8 of an octave, so at most ~12% of the
                          sorted keys are padding and a handful of graphs cover a whole run.
  host:                   .tolist() on the static output        <- the end-of-query sync

Graphs need fixed shapes, so the query is padded to 32 tokens: the padding rows are zero, so
every centroid scores 0 for them, their shifted scores are 0, and Stage 2 drops every task with
a base score <= 0. The candidate set is therefore the same as the eager path's. The filter mask
and the query live in static device buffers that are refilled before each replay.

Not supported (eager path instead): RT-core Stage 1, residual scoring, the dense Stage 3.
"""
from collections import OrderedDict
import numpy as np
import torch

from weft_prep import native_prep

MIN_CAP = 1 << 16
MAX_GRAPHS = 64
MAX_NDOCS = 65536       # candidate buffers are reserved once for this many (never reallocated)


def bucket_cap(P, ndocs, min_cap=MIN_CAP):
    """Smallest capacity >= max(P + 1, ndocs, min_cap) on a grid of 8 steps per power of two."""
    need = max(int(P) + 1, int(ndocs), int(min_cap))
    step = max(1, 1 << max(0, need.bit_length() - 4))     # need in [2^(b-1), 2^b): step = 2^(b-4)
    return ((need + step - 1) // step) * step


def mask_to_i32(mask):
    return int(np.array([mask & 0xFFFFFFFF], dtype=np.uint32).view(np.int32)[0])


class GraphSearch:
    def __init__(self, index, centroids_fp16, top_k, impute=True, reserve=1 << 21):
        self.index, self.C, self.top_k, self.impute = index, centroids_fp16, top_k, impute
        dev = centroids_fp16.device
        self.Qfull = torch.zeros(32, 128, dtype=torch.float16, device=dev)
        self.tokmask = torch.zeros(32, 1, dtype=torch.float16, device=dev)
        self.tok_table = torch.tril(torch.ones(33, 32, dtype=torch.float16, device=dev), diagonal=-1)[:, :, None]
        self.Qsub = torch.zeros(32, 32, 3, dtype=torch.float32, device=dev)   # unused by mode 2 + weight 0
        self.mask_dev = torch.zeros(1, dtype=torch.int32, device=dev)
        self.total = index.total_postings_tensor()
        self.reserve = reserve
        index.reserve_static(reserve, MAX_NDOCS)
        self.known_cap = index.sparse_capacity()
        self.graphs_a, self.graphs_b = {}, OrderedDict()
        self.stats = {"captures_a": 0, "captures_b": 0, "replays": 0, "regrow": 0}

    # -------------------------------------------------------------- capture helpers
    def _capture(self, fn):
        """Warm up on a side stream (cuBLAS workspaces, lazy buffers), then capture fn."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out

    def _ensure_capacity(self, cap):
        """Reserve sparse buffers for cap keys. Any reallocation (here, or by an eager search in the
        same process growing them) invalidates the phase-B graphs, which hold the old pointers."""
        if self.index.sparse_capacity() != self.known_cap:
            self.graphs_b.clear()
        if self.index.sparse_capacity() < cap:
            self.index.reserve_static(max(cap, 2 * self.index.sparse_capacity(), self.reserve), MAX_NDOCS)
            self.graphs_b.clear()
            self.stats["regrow"] += 1
        self.known_cap = self.index.sparse_capacity()

    def _phase_a(self, nc, k_eids):
        key = (nc, k_eids)
        if key not in self.graphs_a:
            def run():
                Qp = self.Qfull * self.tokmask
                topc, scores, _ = native_prep(Qp, self.C, nc, self.impute, compact=True)
                self.index.static_phase_a(self.Qsub, topc, scores, nc, k_eids)
                return (topc, scores)
            self.graphs_a[key] = self._capture(run)
            self.stats["captures_a"] += 1
        return self.graphs_a[key][0]

    def _phase_b(self, cap, ndocs):
        key = (cap, ndocs)
        if key in self.graphs_b:
            self.graphs_b.move_to_end(key)
            return self.graphs_b[key]
        g = self._capture(lambda: self.index.static_phase_b(self.Qfull, cap, ndocs, self.top_k, self.mask_dev))
        self.graphs_b[key] = g
        self.stats["captures_b"] += 1
        while len(self.graphs_b) > MAX_GRAPHS:
            self.graphs_b.popitem(last=False)
        return g

    # -------------------------------------------------------------- search
    @torch.no_grad()
    def search(self, Qf, ntok, nc, ndocs, k_eids, mask=0):
        """Same contract as Engines.ragrt: pids best first (padding dropped, filter applied)."""
        assert ndocs <= MAX_NDOCS
        self.Qfull.copy_(Qf)
        self.tokmask.copy_(self.tok_table[ntok])
        self.mask_dev.fill_(mask_to_i32(mask))
        self._phase_a(nc, k_eids).replay()
        P = int(self.total.item())
        cap = bucket_cap(P, ndocs)
        self._ensure_capacity(cap)
        g, out = self._phase_b(cap, ndocs)
        g.replay()
        self.stats["replays"] += 1
        return [x for x in out.tolist() if x >= 0]
