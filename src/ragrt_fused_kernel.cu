// Stages 1 (no-RT ablations), 2 and 3 of RAGRT.
//
// Stage 2: warp per ray (token x subspace): top-k_eids hit codewords, then for every
//          (codeword, probed centroid) pair a binary search in the CSR row of that centroid,
//          emitting one task per non-empty posting list.
// Stage 3: per passage, sum over query tokens of the max base score of any of its postings.
//          dense : ntok x N fp16 table + atomicMax, then a sum over all N passages.
//          sparse: only the touched postings: one 64-bit key per posting
//                  [pid | token | fp16 score], radix sort, keep the max per (pid, token),
//                  reduce by pid, top-k over the touched passages. No O(N) work.

// cub inside a torch extension: wrap it so it cannot clash with torch's own copy.
#ifndef CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE ragrt_cub
#endif
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>

namespace rcub = ::ragrt_cub::cub;

#define MAX_TASKS (1 << 22)
#define NUM_3D_SUBSPACES 32
#define MIN_BASE_SCORE 0.0f
#define WARPS_PER_BLOCK 8
#define S2_MAX_HITS 256     // must be >= any max_hits passed in (MAX_HITS_CAP)
#define S2_MAX_EIDS 256     // codewords kept per ray (256 = all of an E=256 codebook)

// Diagnostic counters (cumulative until reset; only written when stats are enabled,
// so timed runs are unaffected). Index meaning, also mirrored in ragrt_drop_stats():
#define ST_LAUNCHES        0  // stage-2 launches (= queries)
#define ST_RAYS            1  // rays processed (token x subspace)
#define ST_RAYS_OVERFLOW   2  // rays with more hits than max_hits (best-N kept)
#define ST_HITS_LOST       3  // hits beyond max_hits, summed over rays
#define ST_COMBOS          4  // (kept eid, probed centroid) pairs looked up in the CSR
#define ST_COMBOS_FOUND    5  // ... whose posting list exists
#define ST_MIN_SCORE_DROP  6  // ... found but dropped by MIN_BASE_SCORE
#define ST_TASKS           7  // posting-list tasks emitted (incl. dropped)
#define ST_TASKS_DROPPED   8  // tasks beyond MAX_TASKS
#define ST_HITS            9  // stage-1 hits summed over rays (any-hit work)
#define ST_RAYS_SHORT     10  // rays with fewer hits than k_eids
#define ST_POSTINGS       11  // postings scanned in stage 3
#define ST_COUNT          12

__device__ __forceinline__ void atomicMaxHalf(half* addr, half val) {
#if __CUDA_ARCH__ >= 700
    unsigned short* address_as_us = (unsigned short*)addr;
    unsigned short old = *address_as_us, assumed;
    while (__half_as_ushort(val) > old) {
        assumed = old;
        old = atomicCAS(address_as_us, assumed, __half_as_ushort(val));
        if (assumed == old) break;
    }
#endif
}

// ---------------------------------------------------------------------------
// Stage 2
// ---------------------------------------------------------------------------
template <typename EidT>
__global__ void collect_posting_tasks_csr_warp_kernel(
    const int* __restrict__ out_c, const float* __restrict__ out_v, const int* __restrict__ out_hit_count,
    const int* __restrict__ topc, const float* __restrict__ scores,
    const int64_t* __restrict__ csr_row_ptrs, const EidT* __restrict__ csr_col_eids,
    const int64_t* __restrict__ csr_block_sums, const uint16_t* __restrict__ csr_lengths,
    int total_rays, int max_hits, int k_eids, int k_cent, int ntok, int num_centroids,
    int* __restrict__ d_num_tasks, int* __restrict__ d_num_dropped,
    int* __restrict__ d_task_qt, int64_t* __restrict__ d_task_offset,
    int* __restrict__ d_task_length, float* __restrict__ d_task_base_score,
    unsigned long long* __restrict__ d_task_start, unsigned long long* __restrict__ d_total_postings,
    const int* __restrict__ prim_eid, int E, float cw_weight, int compact_scores,
    unsigned long long* __restrict__ stats
) {
    __shared__ int   s_top_eid[WARPS_PER_BLOCK][S2_MAX_EIDS];
    __shared__ float s_top_val[WARPS_PER_BLOCK][S2_MAX_EIDS];
    __shared__ int   s_hits_eid[WARPS_PER_BLOCK][S2_MAX_HITS];
    __shared__ float s_hits_val[WARPS_PER_BLOCK][S2_MAX_HITS];

    int warp_id       = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id       = threadIdx.x % 32;
    int warp_in_block = threadIdx.x / 32;
    if (warp_id >= total_rays) return;

    int qt = warp_id / NUM_3D_SUBSPACES;
    int s  = warp_id % NUM_3D_SUBSPACES;
    if (qt >= ntok) return;

    int n_hits = out_hit_count[warp_id];
    if (stats && lane_id == 0) {
        if (warp_id == 0) atomicAdd(&stats[ST_LAUNCHES], 1ULL);
        atomicAdd(&stats[ST_RAYS], 1ULL);
        atomicAdd(&stats[ST_HITS], (unsigned long long)n_hits);
        if (n_hits < k_eids) atomicAdd(&stats[ST_RAYS_SHORT], 1ULL);
        if (n_hits > max_hits) {
            atomicAdd(&stats[ST_RAYS_OVERFLOW], 1ULL);
            atomicAdd(&stats[ST_HITS_LOST], (unsigned long long)(n_hits - max_hits));
        }
    }
    if (n_hits > max_hits) n_hits = max_hits;
    int want = (k_eids < n_hits) ? k_eids : n_hits;
    if (want > S2_MAX_EIDS) want = S2_MAX_EIDS;
    int64_t base_hit = (int64_t)warp_id * max_hits;

    // 1. Load the hits into shared memory
    // Polar scene hits are norm-sorted primitive indices: map to codeword ids here
    // (parallel over lanes, instead of a dependent load inside every any-hit call).
    const int* pmap = prim_eid ? prim_eid + (int64_t)s * E : nullptr;
    for (int i = lane_id; i < n_hits; i += 32) {
        int c = out_c[base_hit + i];
        s_hits_eid[warp_in_block][i] = (pmap && c >= 0) ? pmap[c] : c;
        s_hits_val[warp_in_block][i] = out_v[base_hit + i];
    }
    for (int i = n_hits + lane_id; i < S2_MAX_HITS; i += 32) {
        s_hits_eid[warp_in_block][i] = -1;
        s_hits_val[warp_in_block][i] = -1e30f;
    }
    __syncwarp();

    // 2. Top-k by repeated warp argmax; duplicates of the winner (shared triangle edges,
    //    repeated any-hit calls) are removed with it. When every hit is kept there is
    //    nothing to select: copy them (duplicates only repeat a lookup; Stage 3 takes a max).
    int nsel = 0;
    int nchunks = (n_hits + 31) / 32;
    if (want == n_hits) {
        for (int i = lane_id; i < n_hits; i += 32) {
            s_top_eid[warp_in_block][i] = s_hits_eid[warp_in_block][i];
            s_top_val[warp_in_block][i] = s_hits_val[warp_in_block][i];
        }
        nsel = n_hits;
        __syncwarp();
    }
    for (int r = (want == n_hits) ? want : 0; r < want; ++r) {
        float my_best_v = -1e30f;
        int   my_best_i = -1;
        for (int chunk = 0; chunk < nchunks; ++chunk) {
            int i = lane_id + chunk * 32;
            int eid = s_hits_eid[warp_in_block][i];
            float v = s_hits_val[warp_in_block][i];
            if (eid >= 0 && v > my_best_v) { my_best_v = v; my_best_i = i; }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_v = __shfl_down_sync(0xFFFFFFFF, my_best_v, offset);
            int   other_i = __shfl_down_sync(0xFFFFFFFF, my_best_i, offset);
            if (other_v > my_best_v) { my_best_v = other_v; my_best_i = other_i; }
        }
        int winner_i = __shfl_sync(0xFFFFFFFF, my_best_i, 0);
        if (winner_i < 0) break;

        int winner_eid = s_hits_eid[warp_in_block][winner_i];
        if (lane_id == 0) {
            s_top_eid[warp_in_block][nsel] = winner_eid;
            s_top_val[warp_in_block][nsel] = s_hits_val[warp_in_block][winner_i];
        }
        nsel++;
        __syncwarp();
        for (int chunk = 0; chunk < nchunks; ++chunk) {
            int i = lane_id + chunk * 32;
            if (s_hits_eid[warp_in_block][i] == winner_eid) s_hits_eid[warp_in_block][i] = -1;
        }
        __syncwarp();
    }

    // 3. CSR lookups and block-sum offset reconstruction
    int total_combos = nsel * k_cent;
    if (stats && lane_id == 0) atomicAdd(&stats[ST_COMBOS], (unsigned long long)total_combos);
    int64_t s_row_base = (int64_t)s * (num_centroids + 1);

    // Warp-uniform loop (all 32 lanes, full masks) so the scans below are well defined.
    for (int combo0 = 0; combo0 < total_combos; combo0 += 32) {
        int combo = combo0 + lane_id;
        bool valid = combo < total_combos;
        int h = valid ? combo / k_cent : 0;
        int c = valid ? combo % k_cent : 0;
        int eid = s_top_eid[warp_in_block][h];
        int cid = valid ? topc[qt * k_cent + c] : -1;

        int task_found = 0, list_found = 0, score_drop = 0;
        int64_t task_off = 0;
        int task_len = 0;
        float task_base = 0.0f;

        if (cid >= 0 && cid < num_centroids) {
            int64_t lo = csr_row_ptrs[s_row_base + cid];
            int64_t hi = csr_row_ptrs[s_row_base + cid + 1];
            int64_t found_idx = -1;
            while (lo < hi) {
                int64_t mid = lo + ((hi - lo) >> 1);
                int e = (int)csr_col_eids[mid];
                if (e == eid) { found_idx = mid; break; }
                else if (e < eid) lo = mid + 1;
                else hi = mid;
            }
            if (found_idx >= 0) {
                int length = (int)csr_lengths[found_idx];
                if (length > 0) {
                    list_found = 1;
                    // cw_weight = 1: q.c + q_s.w (default). 0: centroid-only scoring (ablation; same lists fetched).
                    // compact_scores: scores is [ntok, k_cent] (aligned with topc) instead of [ntok, num_centroids]
                    float cscore = compact_scores ? scores[qt * k_cent + c] : scores[qt * num_centroids + cid];
                    float base_score = cscore + cw_weight * s_top_val[warp_in_block][h];
                    if (base_score > MIN_BASE_SCORE) {
                        task_found = 1;
                        int64_t b_idx   = found_idx >> 7;
                        int64_t b_start = b_idx << 7;
                        int64_t off = csr_block_sums[b_idx];
                        #pragma unroll 8
                        for (int64_t k = b_start; k < found_idx; ++k) off += (int64_t)csr_lengths[k];
                        task_off = off; task_len = length; task_base = base_score;
                    } else {
                        score_drop = 1;
                    }
                }
            }
        }

        unsigned int mask = __ballot_sync(0xFFFFFFFF, task_found);
        if (stats) {
            unsigned int m_found = __ballot_sync(0xFFFFFFFF, list_found);
            unsigned int m_drop  = __ballot_sync(0xFFFFFFFF, score_drop);
            if (lane_id == 0) {
                if (m_found) atomicAdd(&stats[ST_COMBOS_FOUND], (unsigned long long)__popc(m_found));
                if (m_drop)  atomicAdd(&stats[ST_MIN_SCORE_DROP], (unsigned long long)__popc(m_drop));
                if (mask)    atomicAdd(&stats[ST_TASKS], (unsigned long long)__popc(mask));
            }
        }
        if (!mask) continue;   // warp-uniform

        int task_base_idx = 0;
        if (lane_id == 0) task_base_idx = atomicAdd(d_num_tasks, __popc(mask));
        task_base_idx = __shfl_sync(0xFFFFFFFF, task_base_idx, 0);
        int my_task_idx = task_base_idx + __popc(mask & ((1u << lane_id) - 1));
        bool stored = task_found && my_task_idx < MAX_TASKS;

        // Posting start of each stored task (sparse stage 3): warp prefix sum, one atomic.
        unsigned long long len = stored ? (unsigned long long)task_len : 0ULL;
        unsigned long long incl = len;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            unsigned long long o = __shfl_up_sync(0xFFFFFFFF, incl, d);
            if (lane_id >= d) incl += o;
        }
        unsigned long long warp_len = __shfl_sync(0xFFFFFFFF, incl, 31);
        unsigned long long post_base = 0;
        if (lane_id == 31 && warp_len) post_base = atomicAdd(d_total_postings, warp_len);
        post_base = __shfl_sync(0xFFFFFFFF, post_base, 31);

        if (stored) {
            d_task_qt[my_task_idx]         = qt;
            d_task_offset[my_task_idx]     = task_off;
            d_task_length[my_task_idx]     = task_len;
            d_task_base_score[my_task_idx] = task_base;
            d_task_start[my_task_idx]      = post_base + incl - len;
        } else if (task_found) {
            atomicAdd(d_num_dropped, 1);
            if (stats) atomicAdd(&stats[ST_TASKS_DROPPED], 1ULL);
        }
    }
}

__global__ void add_postings_stat_kernel(const unsigned long long* total, unsigned long long* stats) {
    atomicAdd(&stats[ST_POSTINGS], *total);
}

// ---------------------------------------------------------------------------
// Stage 3, dense
// ---------------------------------------------------------------------------
__global__ void cooperative_score_passages_csr_fp16_filtered_kernel(
    const int* __restrict__ d_num_tasks, int* __restrict__ d_queue_counter,
    const int* __restrict__ d_task_qt, const int64_t* __restrict__ d_task_offset,
    const int* __restrict__ d_task_length, const float* __restrict__ d_task_base_score,
    const uint8_t* __restrict__ map_packed_24, int num_passages, half* __restrict__ qt_pid_max_fp16,
    const uint32_t* __restrict__ doc_predicates, uint32_t query_mask
) {
    int total_tasks = *d_num_tasks;
    if (total_tasks > MAX_TASKS) total_tasks = MAX_TASKS;
    int lane_id = threadIdx.x % 32;

    while (true) {
        int task_id = 0;
        if (lane_id == 0) task_id = atomicAdd(d_queue_counter, 1);
        task_id = __shfl_sync(0xFFFFFFFF, task_id, 0);
        if (task_id >= total_tasks) break;

        int qt           = d_task_qt[task_id];
        int64_t offset   = d_task_offset[task_id];
        int length       = d_task_length[task_id];
        half base_score_h = __float2half(d_task_base_score[task_id]);

        int prev_pid = -1;
        for (int chunk = 0; chunk < length; chunk += 32) {
            int idx = chunk + lane_id;
            int pid = -1;
            if (idx < length) {
                const uint8_t* p24 = map_packed_24 + (offset + idx) * 3;
                pid = p24[0] | (p24[1] << 8) | (p24[2] << 16);
            }
            int left_pid = __shfl_up_sync(0xFFFFFFFF, pid, 1);
            bool dup = (pid >= 0) &&
                       ((lane_id > 0 && pid == left_pid) || (lane_id == 0 && pid == prev_pid));
            if (idx < length && !dup && pid >= 0 && pid < num_passages) {
                if (query_mask == 0 || (doc_predicates[pid] & query_mask) != 0) {
                    atomicMaxHalf(&qt_pid_max_fp16[qt * num_passages + pid], base_score_h);
                }
            }
            prev_pid = __shfl_sync(0xFFFFFFFF, pid, 31);
        }
    }
}

__global__ void sum_passages_fp16_to_fp32_kernel(
    const half* __restrict__ qt_pid_max_fp16, int ntok, int num_passages, float* __restrict__ passage_scores,
    const uint32_t* __restrict__ doc_predicates, uint32_t query_mask
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_passages) return;
    if (query_mask != 0 && (doc_predicates[pid] & query_mask) == 0) {
        passage_scores[pid] = -1e30f;
        return;
    }
    float total = 0.0f;
    #pragma unroll 8
    for (int q = 0; q < ntok; ++q) {
        float val = __half2float(qt_pid_max_fp16[q * num_passages + pid]);
        if (val > 0.0f) total += val;
    }
    passage_scores[pid] = total;
}

// ---------------------------------------------------------------------------
// Stage 3, sparse
// key = pid << 20 | token << 15 | fp16 bits of the (positive) base score.
// Sorted keys: the last element of each (pid, token) run holds that token's max.
// ---------------------------------------------------------------------------
#define SP_SCORE_BITS 15
#define SP_TOK_SHIFT  15
#define SP_PID_SHIFT  20

__global__ void sparse_expand_kernel(
    const int* __restrict__ d_num_tasks,
    const int* __restrict__ d_task_qt, const int64_t* __restrict__ d_task_offset,
    const int* __restrict__ d_task_length, const float* __restrict__ d_task_base_score,
    const unsigned long long* __restrict__ d_task_start, long long cap,
    const uint8_t* __restrict__ map_packed_24, int num_passages,
    const uint32_t* __restrict__ doc_predicates, uint32_t query_mask,
    unsigned long long sentinel, unsigned long long* __restrict__ keys,
    const uint32_t* __restrict__ tok_ids, const uint8_t* __restrict__ residuals, const float* __restrict__ lut,
    const uint32_t* __restrict__ d_mask
) {
    if (d_mask) query_mask = *d_mask;     // static (graph) path: the filter lives in device memory
    int total_tasks = *d_num_tasks;
    if (total_tasks > MAX_TASKS) total_tasks = MAX_TASKS;
    int lane = threadIdx.x % 32;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int nwarps = (gridDim.x * blockDim.x) / 32;
    for (int t = warp; t < total_tasks; t += nwarps) {
        int qt = d_task_qt[t];
        int64_t off = d_task_offset[t];
        int len = d_task_length[t];
        unsigned long long start = d_task_start[t];
        float base = d_task_base_score[t];
        unsigned long long sbits = (unsigned long long)(__half_as_ushort(__float2half(base)) & 0x7FFF);
        unsigned long long tok = ((unsigned long long)qt) << SP_TOK_SHIFT;
        const float* L = lut ? lut + (size_t)qt * 32 * 256 : nullptr;
        for (int i = lane; i < len; i += 32) {
            long long dst = (long long)start + i;
            if (dst >= cap) break;
            const uint8_t* p24 = map_packed_24 + (off + i) * 3;
            int pid = p24[0] | (p24[1] << 8) | (p24[2] << 16);
            if (L) {   // per-token score: centroid part + residual part, clamped at the imputed floor (0)
                const uint4* r4 = reinterpret_cast<const uint4*>(residuals + (size_t)tok_ids[off + i] * 32);
                uint4 a = r4[0], b2 = r4[1];
                const uint8_t* rb = reinterpret_cast<const uint8_t*>(&a);
                const uint8_t* rc = reinterpret_cast<const uint8_t*>(&b2);
                float sc = base;
                #pragma unroll
                for (int j = 0; j < 16; ++j) sc += L[j * 256 + rb[j]];
                #pragma unroll
                for (int j = 0; j < 16; ++j) sc += L[(16 + j) * 256 + rc[j]];
                sbits = (unsigned long long)(__half_as_ushort(__float2half(fmaxf(sc, 0.0f))) & 0x7FFF);
            }
            unsigned long long k = sentinel;
            if (pid < num_passages && (query_mask == 0 || (doc_predicates[pid] & query_mask) != 0))
                k = (((unsigned long long)pid) << SP_PID_SHIFT) | tok | sbits;
            keys[dst] = k;
        }
    }
}

// Static path: keys [P, cap) become the sentinel, so a sort over all cap keys puts them last.
__global__ void sparse_fill_sentinel_kernel(const unsigned long long* __restrict__ d_total, long long cap,
                                            unsigned long long sentinel, unsigned long long* __restrict__ keys) {
    long long P = (long long)*d_total;
    if (P > cap) P = cap;
    for (long long i = P + (long long)blockIdx.x * blockDim.x + threadIdx.x; i < cap; i += (long long)gridDim.x * blockDim.x)
        keys[i] = sentinel;
}

__global__ void sparse_tail_values_kernel(
    const unsigned long long* __restrict__ keys, long long n, int num_passages,
    int* __restrict__ pid_out, float* __restrict__ val_out
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long k = keys[i];
    int pid = (int)(k >> SP_PID_SHIFT);
    bool tail = (i == n - 1) || ((keys[i + 1] >> SP_TOK_SHIFT) != (k >> SP_TOK_SHIFT));
    pid_out[i] = pid;
    float v = 0.0f;
    if (tail && pid < num_passages)
        v = __half2float(__ushort_as_half((unsigned short)(k & 0x7FFF)));
    val_out[i] = v;
}

// Entries [num_runs, L) and the sentinel run (pid >= N) become (-1, -inf).
__global__ void sparse_fixup_kernel(
    const int* __restrict__ d_num_runs, long long L, int num_passages,
    int* __restrict__ pids, float* __restrict__ sums
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;
    int nr = *d_num_runs;
    if (i >= nr || pids[i] >= num_passages || pids[i] < 0) {
        pids[i] = -1;
        sums[i] = -INFINITY;
    }
}

// ---------------------------------------------------------------------------
// Stage 3, hash (no sort). Table 1: open addressing over 64-bit slots
// [key + 1 (48 bits) | fp16 score bits (16)], key = pid * 32 + token: keeps the max score per
// (passage, token). Table 2: pid + 1 -> float sum of those maxima. Failing (filtered) postings
// never enter the tables. Both tables fit in L2 at the sizes WEFT uses (a few MB to ~30 MB).
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long hash_slot(unsigned long long key, int log2cap) {
    return (key * 0x9E3779B97F4A7C15ULL) >> (64 - log2cap);
}

__global__ void hash_insert_kernel(
    const int* __restrict__ d_num_tasks,
    const int* __restrict__ d_task_qt, const int64_t* __restrict__ d_task_offset,
    const int* __restrict__ d_task_length, const float* __restrict__ d_task_base_score,
    const uint8_t* __restrict__ map_packed_24, int num_passages,
    const uint32_t* __restrict__ doc_predicates, uint32_t query_mask,
    unsigned long long* __restrict__ table, int log2cap,
    const uint32_t* __restrict__ tok_ids, const uint8_t* __restrict__ residuals, const float* __restrict__ lut
) {
    int total_tasks = *d_num_tasks;
    if (total_tasks > MAX_TASKS) total_tasks = MAX_TASKS;
    int lane = threadIdx.x % 32;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int nwarps = (gridDim.x * blockDim.x) / 32;
    unsigned long long cmask = (1ULL << log2cap) - 1ULL;
    for (int t = warp; t < total_tasks; t += nwarps) {
        int qt = d_task_qt[t];
        int64_t off = d_task_offset[t];
        int len = d_task_length[t];
        float base = d_task_base_score[t];
        unsigned long long sbits = (unsigned long long)(__half_as_ushort(__float2half(base)) & 0x7FFF);
        const float* L = lut ? lut + (size_t)qt * 32 * 256 : nullptr;
        for (int i = lane; i < len; i += 32) {
            const uint8_t* p24 = map_packed_24 + (off + i) * 3;
            int pid = p24[0] | (p24[1] << 8) | (p24[2] << 16);
            if (pid >= num_passages || (query_mask != 0 && (doc_predicates[pid] & query_mask) == 0)) continue;
            unsigned long long sb = sbits;
            if (L) {
                const uint4* r4 = reinterpret_cast<const uint4*>(residuals + (size_t)tok_ids[off + i] * 32);
                uint4 a = r4[0], b2 = r4[1];
                const uint8_t* rb = reinterpret_cast<const uint8_t*>(&a);
                const uint8_t* rc = reinterpret_cast<const uint8_t*>(&b2);
                float sc = base;
                #pragma unroll
                for (int j = 0; j < 16; ++j) sc += L[j * 256 + rb[j]];
                #pragma unroll
                for (int j = 0; j < 16; ++j) sc += L[(16 + j) * 256 + rc[j]];
                sb = (unsigned long long)(__half_as_ushort(__float2half(fmaxf(sc, 0.0f))) & 0x7FFF);
            }
            unsigned long long key = ((unsigned long long)pid << 5) | (unsigned long long)qt;
            unsigned long long tag = (key + 1ULL) << 16;
            unsigned long long h = hash_slot(key, log2cap);
            while (true) {
                unsigned long long old = table[h];
                if (old == 0ULL) {
                    old = atomicCAS(&table[h], 0ULL, tag | sb);
                    if (old == 0ULL) break;
                }
                if ((old & ~0xFFFFULL) == tag) {
                    while ((old & 0xFFFFULL) < sb) {
                        unsigned long long seen = atomicCAS(&table[h], old, tag | sb);
                        if (seen == old) break;
                        old = seen;
                    }
                    break;
                }
                h = (h + 1) & cmask;
            }
        }
    }
}

__global__ void hash_reduce_kernel(
    const unsigned long long* __restrict__ table, long long cap, int log2cap,
    int* __restrict__ pid_keys, float* __restrict__ pid_sums
) {
    unsigned long long cmask = (1ULL << log2cap) - 1ULL;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < cap; i += (long long)gridDim.x * blockDim.x) {
        unsigned long long s = table[i];
        if (s == 0ULL) continue;
        unsigned long long key = (s >> 16) - 1ULL;
        int pid = (int)(key >> 5);
        float v = __half2float(__ushort_as_half((unsigned short)(s & 0x7FFFULL)));
        unsigned long long h = hash_slot((unsigned long long)pid, log2cap);
        while (true) {
            int old = pid_keys[h];
            if (old == 0) old = atomicCAS(&pid_keys[h], 0, pid + 1);
            if (old == 0 || old == pid + 1) { atomicAdd(&pid_sums[h], v); break; }
            h = (h + 1) & cmask;
        }
    }
}

__global__ void hash_compact_kernel(
    const int* __restrict__ pid_keys, const float* __restrict__ pid_sums, long long cap,
    int* __restrict__ d_count, int* __restrict__ pids_out, float* __restrict__ sums_out
) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < cap; i += (long long)gridDim.x * blockDim.x) {
        int k = pid_keys[i];
        if (k == 0) continue;
        int j = atomicAdd(d_count, 1);
        pids_out[j] = k - 1;
        sums_out[j] = pid_sums[i];
    }
}

// ---------------------------------------------------------------------------
// Stage 3, 32-bit pairs: key = pid << 5 | token (uint32), value = fp16 score bits (uint16).
// About 4 radix passes over 6-byte items instead of ~6 over 8-byte keys; the max per
// (pid, token) becomes a reduce-by-key (scores are no longer ordered inside a run).
// ---------------------------------------------------------------------------
#define K32_SENTINEL 0xFFFFFFFFu

__global__ void sparse_expand32_kernel(
    const int* __restrict__ d_num_tasks,
    const int* __restrict__ d_task_qt, const int64_t* __restrict__ d_task_offset,
    const int* __restrict__ d_task_length, const float* __restrict__ d_task_base_score,
    const unsigned long long* __restrict__ d_task_start, long long cap,
    const uint8_t* __restrict__ map_packed_24, int num_passages,
    const uint32_t* __restrict__ doc_predicates, uint32_t query_mask,
    uint32_t* __restrict__ keys, uint16_t* __restrict__ vals,
    const uint32_t* __restrict__ tok_ids, const uint8_t* __restrict__ residuals, const float* __restrict__ lut
) {
    int total_tasks = *d_num_tasks;
    if (total_tasks > MAX_TASKS) total_tasks = MAX_TASKS;
    int lane = threadIdx.x % 32;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int nwarps = (gridDim.x * blockDim.x) / 32;
    for (int t = warp; t < total_tasks; t += nwarps) {
        int qt = d_task_qt[t];
        int64_t off = d_task_offset[t];
        int len = d_task_length[t];
        unsigned long long start = d_task_start[t];
        float base = d_task_base_score[t];
        uint16_t sbits = (uint16_t)(__half_as_ushort(__float2half(base)) & 0x7FFF);
        const float* L = lut ? lut + (size_t)qt * 32 * 256 : nullptr;
        for (int i = lane; i < len; i += 32) {
            long long dst = (long long)start + i;
            if (dst >= cap) break;
            const uint8_t* p24 = map_packed_24 + (off + i) * 3;
            int pid = p24[0] | (p24[1] << 8) | (p24[2] << 16);
            uint16_t sb = sbits;
            if (L) {
                const uint4* r4 = reinterpret_cast<const uint4*>(residuals + (size_t)tok_ids[off + i] * 32);
                uint4 a = r4[0], b2 = r4[1];
                const uint8_t* rb = reinterpret_cast<const uint8_t*>(&a);
                const uint8_t* rc = reinterpret_cast<const uint8_t*>(&b2);
                float sc = base;
                #pragma unroll
                for (int j = 0; j < 16; ++j) sc += L[j * 256 + rb[j]];
                #pragma unroll
                for (int j = 0; j < 16; ++j) sc += L[(16 + j) * 256 + rc[j]];
                sb = (uint16_t)(__half_as_ushort(__float2half(fmaxf(sc, 0.0f))) & 0x7FFF);
            }
            bool ok = pid < num_passages && (query_mask == 0 || (doc_predicates[pid] & query_mask) != 0);
            keys[dst] = ok ? (((uint32_t)pid << 5) | (uint32_t)qt) : K32_SENTINEL;
            vals[dst] = ok ? sb : (uint16_t)0;
        }
    }
}

// (pid, token) runs -> (pid, float max). Entries past the run count, and the sentinel run, get an
// out-of-range pid (the final fixup turns them into padding).
__global__ void pairs_to_pid_kernel(
    const uint32_t* __restrict__ ukeys, const uint16_t* __restrict__ umax, const int* __restrict__ d_runs,
    long long n, int* __restrict__ pid_out, float* __restrict__ val_out
) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = *d_runs;
    if (i < r && ukeys[i] != K32_SENTINEL) {
        pid_out[i] = (int)(ukeys[i] >> 5);
        val_out[i] = __half2float(__ushort_as_half(umax[i]));
    } else {
        pid_out[i] = 0x7FFFFFFF;
        val_out[i] = 0.0f;
    }
}

struct MaxOpU16 {
    __device__ __forceinline__ uint16_t operator()(const uint16_t& a, const uint16_t& b) const { return a > b ? a : b; }
};

struct SumOp {
    __device__ __forceinline__ float operator()(const float& a, const float& b) const { return a + b; }
};

// After TileMaxSim: candidates that fail the filter score -inf, so they rank last.
__global__ void mask_failing_candidates_kernel(
    const int* __restrict__ cand_pids, int n, const uint32_t* __restrict__ doc_predicates,
    uint32_t query_mask, float* __restrict__ scores, const uint32_t* __restrict__ d_mask
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (d_mask) { query_mask = *d_mask; if (query_mask == 0) return; }
    int pid = cand_pids[i];
    if (pid < 0 || (doc_predicates[pid] & query_mask) == 0) scores[i] = -INFINITY;
}

extern "C" void launch_mask_failing_candidates(
    const int* d_cand_pids, int n, const uint32_t* d_doc_predicates, uint32_t query_mask,
    float* d_scores, cudaStream_t stream
) {
    if (query_mask == 0 || n <= 0) return;
    int threads = 256;
    mask_failing_candidates_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        d_cand_pids, n, d_doc_predicates, query_mask, d_scores, nullptr);
}

// Same, with the filter read from device memory (static / CUDA-graph path).
extern "C" void launch_mask_failing_candidates_dev(
    const int* d_cand_pids, int n, const uint32_t* d_doc_predicates, const uint32_t* d_mask,
    float* d_scores, cudaStream_t stream
) {
    if (n <= 0) return;
    int threads = 256;
    mask_failing_candidates_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        d_cand_pids, n, d_doc_predicates, 0u, d_scores, d_mask);
}

// ---------------------------------------------------------------------------
// Stage 1 WITHOUT RT cores (ablations). Ray = token * 32 + subspace, same layout and
// output buffers as the OptiX raygen, so Stages 2-4 are unchanged.
//
// (a) exact top-k (fan geometry's reference, and tests): k rounds of a warp argmax.
//     Cost k * E / 32 dots per lane, fine for tests, slow for large E.
// (b) threshold scan (ablation of the polar geometry): one pass over E, keeps every
//     codeword with d.c >= tau_s, exactly what the polar RT scene returns. Codewords are
//     staged through shared memory, shared by the warps (tokens) of one subspace.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool bf_before(float va, int ia, float vb, int ib) {
    return (va > vb) || (va == vb && ia < ib);
}

__global__ void bruteforce_stage1_kernel(
    const float* __restrict__ q_points, const float* __restrict__ codewords,
    int total_rays, int E, int k, int max_hits,
    int* __restrict__ out_c, float* __restrict__ out_v, int* __restrict__ out_n
) {
    int ray  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (ray >= total_rays) return;
    int s = ray % NUM_3D_SUBSPACES;
    float qx = q_points[3 * ray], qy = q_points[3 * ray + 1], qz = q_points[3 * ray + 2];
    const float* cw = codewords + (size_t)3 * s * E;
    if (k > max_hits) k = max_hits;

    float last_v = INFINITY; int last_i = -1;
    int nsel = 0;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY; int bi = -1;
        for (int e = lane; e < E; e += 32) {
            float v = qx * cw[3 * e] + qy * cw[3 * e + 1] + qz * cw[3 * e + 2];
            if (!(v > 0.0f)) continue;
            if (!bf_before(last_v, last_i, v, e)) continue;
            if (bi < 0 || bf_before(v, e, bv, bi)) { bv = v; bi = e; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off /= 2) {
            float ov = __shfl_down_sync(0xFFFFFFFF, bv, off);
            int   oi = __shfl_down_sync(0xFFFFFFFF, bi, off);
            if (oi >= 0 && (bi < 0 || bf_before(ov, oi, bv, bi))) { bv = ov; bi = oi; }
        }
        bv = __shfl_sync(0xFFFFFFFF, bv, 0);
        bi = __shfl_sync(0xFFFFFFFF, bi, 0);
        if (bi < 0) break;
        if (lane == 0) {
            out_c[(size_t)ray * max_hits + nsel] = bi;
            out_v[(size_t)ray * max_hits + nsel] = bv;
        }
        last_v = bv; last_i = bi;
        ++nsel;
    }
    if (lane == 0) out_n[ray] = nsel;
}

extern "C" void launch_bruteforce_stage1(
    const float* d_q_points, const float* d_codewords, int total_rays, int E, int k, int max_hits,
    int* d_out_c, float* d_out_v, int* d_out_n, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_rays * 32 + threads - 1) / threads;
    bruteforce_stage1_kernel<<<blocks, threads, 0, stream>>>(
        d_q_points, d_codewords, total_rays, E, k, max_hits, d_out_c, d_out_v, d_out_n);
}

// (c) every codeword (ablation "codebook filter off"): Stage 2 then fetches every list of
//     each probed centroid. Requires E <= max_hits.
__global__ void all_codewords_stage1_kernel(
    const float* __restrict__ q_points, const float* __restrict__ codewords,
    int E, int max_hits, int* __restrict__ out_c, float* __restrict__ out_v, int* __restrict__ out_n
) {
    int ray = blockIdx.x;
    int s = ray % NUM_3D_SUBSPACES;
    float qx = q_points[3 * ray], qy = q_points[3 * ray + 1], qz = q_points[3 * ray + 2];
    const float* cw = codewords + (size_t)3 * s * E;
    size_t base = (size_t)ray * max_hits;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        out_c[base + e] = e;
        out_v[base + e] = qx * cw[3 * e] + qy * cw[3 * e + 1] + qz * cw[3 * e + 2];
    }
    if (threadIdx.x == 0) out_n[ray] = E;
}

extern "C" void launch_all_codewords_stage1(
    const float* d_q_points, const float* d_codewords, int total_rays, int E, int max_hits,
    int* d_out_c, float* d_out_v, int* d_out_n, cudaStream_t stream
) {
    all_codewords_stage1_kernel<<<total_rays, 128, 0, stream>>>(d_q_points, d_codewords, E, max_hits, d_out_c, d_out_v, d_out_n);
}

#define BFT_WARPS 8
#define BFT_CHUNK 1024

__global__ void threshold_stage1_kernel(
    const float* __restrict__ q_points, const float* __restrict__ codewords,
    const float* __restrict__ tau, int ntok, int E, int max_hits,
    int* __restrict__ out_c, float* __restrict__ out_v, int* __restrict__ out_n
) {
    __shared__ float s_cw[BFT_CHUNK * 3];
    int s = blockIdx.x % NUM_3D_SUBSPACES;
    int tok_block = blockIdx.x / NUM_3D_SUBSPACES;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int tok = tok_block * BFT_WARPS + warp;
    bool active = tok < ntok;
    int ray = tok * NUM_3D_SUBSPACES + s;

    float qx = 0.f, qy = 0.f, qz = 0.f, thr = INFINITY;
    if (active) {
        qx = q_points[3 * ray]; qy = q_points[3 * ray + 1]; qz = q_points[3 * ray + 2];
        thr = tau[s] * sqrtf(qx * qx + qy * qy + qz * qz);   // q.c >= tau |q|  <=>  d.c >= tau
    }
    int count = 0;
    size_t base = (size_t)ray * max_hits;
    const float* cw = codewords + (size_t)3 * s * E;

    for (int c0 = 0; c0 < E; c0 += BFT_CHUNK) {
        int n = min(BFT_CHUNK, E - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < 3 * n; i += blockDim.x) s_cw[i] = cw[3 * c0 + i];
        __syncthreads();
        if (!active) continue;
        for (int j0 = 0; j0 < n; j0 += 32) {
            int j = j0 + lane;
            float v = -INFINITY;
            if (j < n) v = qx * s_cw[3 * j] + qy * s_cw[3 * j + 1] + qz * s_cw[3 * j + 2];
            bool pass = (j < n) && (v >= thr) && (v > 0.0f);
            unsigned int m = __ballot_sync(0xFFFFFFFF, pass);
            if (!m) continue;
            int slot = count + __popc(m & ((1u << lane) - 1));
            if (pass && slot < max_hits) {
                out_c[base + slot] = c0 + j;
                out_v[base + slot] = v;
            }
            int old = count;
            count += __popc(m);
            if (count > max_hits) {
                // Overflow: insert the remaining hits one by one, replacing the minimum.
                __syncwarp();
                unsigned int over = 0;
                unsigned int mm = m;
                while (mm) {
                    int src = __ffs(mm) - 1; mm &= mm - 1;
                    int sl = old + __popc(m & ((1u << src) - 1));
                    if (sl >= max_hits) over |= (1u << src);
                }
                while (over) {
                    int src = __ffs(over) - 1; over &= over - 1;
                    float hv = __shfl_sync(0xFFFFFFFF, v, src);
                    int   he = c0 + j0 + src;
                    float mv = INFINITY; int mi = -1;
                    for (int t = lane; t < max_hits; t += 32) {
                        float x = out_v[base + t];
                        if (x < mv) { mv = x; mi = t; }
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off /= 2) {
                        float ov = __shfl_down_sync(0xFFFFFFFF, mv, off);
                        int   oi = __shfl_down_sync(0xFFFFFFFF, mi, off);
                        if (ov < mv) { mv = ov; mi = oi; }
                    }
                    mv = __shfl_sync(0xFFFFFFFF, mv, 0);
                    mi = __shfl_sync(0xFFFFFFFF, mi, 0);
                    if (lane == 0 && hv > mv) { out_c[base + mi] = he; out_v[base + mi] = hv; }
                    __syncwarp();
                }
            }
        }
    }
    if (active && lane == 0) out_n[ray] = count;
}

extern "C" void launch_threshold_stage1(
    const float* d_q_points, const float* d_codewords, const float* d_tau, int total_rays, int E,
    int max_hits, int* d_out_c, float* d_out_v, int* d_out_n, cudaStream_t stream
) {
    int ntok = total_rays / NUM_3D_SUBSPACES;
    int tok_blocks = (ntok + BFT_WARPS - 1) / BFT_WARPS;
    threshold_stage1_kernel<<<tok_blocks * NUM_3D_SUBSPACES, BFT_WARPS * 32, 0, stream>>>(
        d_q_points, d_codewords, d_tau, ntok, E, max_hits, d_out_c, d_out_v, d_out_n);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
static unsigned long long* g_d_stats = nullptr;
static bool g_stats_enabled = false;

static unsigned long long* stats_ptr() {
    if (!g_stats_enabled) return nullptr;
    if (!g_d_stats) {
        cudaMalloc(&g_d_stats, ST_COUNT * sizeof(unsigned long long));
        cudaMemset(g_d_stats, 0, ST_COUNT * sizeof(unsigned long long));
        cudaDeviceSynchronize();
    }
    return g_d_stats;
}

extern "C" void ragrt_set_drop_stats(int enabled) { g_stats_enabled = (enabled != 0); }

// Weight of the codeword term in each posting's Stage 2 score (ablation knob; default 1).
static float g_cw_weight = 1.0f;
extern "C" void ragrt_set_codeword_weight(float w) { g_cw_weight = w; }
extern "C" float ragrt_get_codeword_weight() { return g_cw_weight; }

// Stage 3 implementation for ragrt_stage3_sparse: 0 = 64-bit key radix sort (default),
// 1 = hash tables, 2 = 32-bit key / 16-bit value pair sort.
static int g_stage3_mode = 0;
extern "C" void ragrt_set_stage3_hash(int on) { g_stage3_mode = on ? 1 : 0; }
extern "C" int ragrt_get_stage3_hash() { return g_stage3_mode == 1; }
extern "C" void ragrt_set_stage3_mode(int mode) { g_stage3_mode = (mode >= 0 && mode <= 2) ? mode : 0; }
extern "C" int ragrt_get_stage3_mode() { return g_stage3_mode; }

// Compact centroid scores: Stage 2 reads scores[qt * k_cent + c] (only the probed centroids' scores,
// aligned with topc) instead of the full [ntok, num_centroids] matrix. Saves building that matrix.
static int g_compact_scores = 0;
extern "C" void ragrt_set_compact_scores(int on) { g_compact_scores = on ? 1 : 0; }
extern "C" int ragrt_get_compact_scores() { return g_compact_scores; }

// Residual scoring (WARP-style implicit decompression) in sparse Stage 3: each posting is one
// token (token-level plain-IVF index, csr_token_ids), scored base + sum_j lut[qt][j][residual byte j],
// where lut[qt][j][b] = sum_k q[qt][4j+k] * bucket_weight[decomp[rbm[b]][k]] (built in Python).
// All three pointers non-null = on.
static const uint32_t* g_tok_ids = nullptr;
static const uint8_t*  g_res = nullptr;
static const float*    g_res_lut = nullptr;
extern "C" void ragrt_set_residual_scoring(const uint32_t* tok_ids, const uint8_t* residuals, const float* lut) {
    g_tok_ids = tok_ids; g_res = residuals; g_res_lut = lut;
}
extern "C" int ragrt_residual_scoring_on() { return (g_tok_ids && g_res && g_res_lut) ? 1 : 0; }

extern "C" void ragrt_reset_drop_stats() {
    if (g_d_stats) {
        cudaDeviceSynchronize();
        cudaMemset(g_d_stats, 0, ST_COUNT * sizeof(unsigned long long));
        cudaDeviceSynchronize();
    }
}

extern "C" int ragrt_num_drop_stats() { return ST_COUNT; }

extern "C" void ragrt_drop_stats(unsigned long long* out) {
    for (int i = 0; i < ST_COUNT; ++i) out[i] = 0ULL;
    if (!g_d_stats) return;
    cudaDeviceSynchronize();
    cudaMemcpy(out, g_d_stats, ST_COUNT * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
}

// ---------------------------------------------------------------------------
// Buffers (two sets, for the dual-stream pipeline)
// ---------------------------------------------------------------------------
struct Bufs {
    bool init = false;
    int *num_tasks = nullptr, *queue_counter = nullptr, *num_dropped = nullptr;
    int *task_qt = nullptr, *task_length = nullptr;
    int64_t* task_offset = nullptr;
    float* task_base = nullptr;
    unsigned long long *task_start = nullptr, *total_postings = nullptr;
    unsigned long long* h_total = nullptr;                // pinned
    half* qt_pid_max = nullptr; int dense_passages = 0;   // dense stage 3 (lazy)
    // sparse stage 3 (grown on demand)
    long long sp_cap = 0;
    unsigned long long *keys_a = nullptr, *keys_b = nullptr;
    int *run_pid_in = nullptr, *run_pid_out = nullptr, *num_runs = nullptr, *num_runs1 = nullptr;
    float *val_in = nullptr, *sums = nullptr;
    void* temp = nullptr; size_t temp_bytes = 0;
};
static Bufs g_b[2];

static void ensure_task_bufs(int b) {
    Bufs& B = g_b[b];
    if (B.init) return;
    cudaMalloc(&B.num_tasks, sizeof(int));
    cudaMalloc(&B.queue_counter, sizeof(int));
    cudaMalloc(&B.num_dropped, sizeof(int));
    cudaMalloc(&B.task_qt, MAX_TASKS * sizeof(int));
    cudaMalloc(&B.task_offset, MAX_TASKS * sizeof(int64_t));
    cudaMalloc(&B.task_length, MAX_TASKS * sizeof(int));
    cudaMalloc(&B.task_base, MAX_TASKS * sizeof(float));
    cudaMalloc(&B.task_start, MAX_TASKS * sizeof(unsigned long long));
    cudaMalloc(&B.total_postings, sizeof(unsigned long long));
    cudaMalloc(&B.num_runs, sizeof(int));
    cudaMalloc(&B.num_runs1, sizeof(int));
    cudaMallocHost(&B.h_total, sizeof(unsigned long long));
    B.init = true;
}

static void ensure_dense(int b, int num_passages) {
    Bufs& B = g_b[b];
    if (B.qt_pid_max && B.dense_passages == num_passages) return;
    if (B.qt_pid_max) cudaFree(B.qt_pid_max);
    cudaMalloc(&B.qt_pid_max, (size_t)32 * num_passages * sizeof(half));
    B.dense_passages = num_passages;
}

static void ensure_sparse(int b, long long need, cudaStream_t stream) {
    Bufs& B = g_b[b];
    if (need > B.sp_cap) {
        long long cap = need + need / 2 + 65536;
        cudaStreamSynchronize(stream);
        cudaFree(B.keys_a); cudaFree(B.keys_b); cudaFree(B.run_pid_in); cudaFree(B.run_pid_out);
        cudaFree(B.val_in); cudaFree(B.sums); cudaFree(B.temp);
        cudaMalloc(&B.keys_a, cap * sizeof(unsigned long long));
        cudaMalloc(&B.keys_b, cap * sizeof(unsigned long long));
        cudaMalloc(&B.run_pid_in, cap * sizeof(int));
        cudaMalloc(&B.run_pid_out, cap * sizeof(int));
        cudaMalloc(&B.val_in, cap * sizeof(float));
        cudaMalloc(&B.sums, cap * sizeof(float));
        size_t t1 = 0, t2 = 0;
        rcub::DeviceRadixSort::SortKeys(nullptr, t1, B.keys_a, B.keys_b, (int)cap, 0, 64, stream);
        rcub::DeviceReduce::ReduceByKey(nullptr, t2, B.run_pid_in, B.run_pid_out, B.val_in, B.sums,
                                        B.num_runs, SumOp(), (int)cap, stream);
        size_t t3 = 0, t4 = 0;     // 32-bit pair path (keys_a / keys_b reinterpreted: 4 B keys + 2 B values)
        uint32_t* k32 = reinterpret_cast<uint32_t*>(B.keys_a);
        uint16_t* v16 = reinterpret_cast<uint16_t*>(k32 + cap);
        rcub::DeviceRadixSort::SortPairs(nullptr, t3, k32, k32, v16, v16, (int)cap, 0, 32, stream);
        rcub::DeviceReduce::ReduceByKey(nullptr, t4, k32, k32, v16, v16, B.num_runs1, MaxOpU16(), (int)cap, stream);
        B.temp_bytes = std::max(std::max(t1, t2), std::max(t3, t4));
        cudaMalloc(&B.temp, B.temp_bytes);
        B.sp_cap = cap;
    }
}

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------
extern "C" void ragrt_stage2(
    const int* d_out_c, const float* d_out_v, const int* d_out_hit_count,
    const int* d_topc, const float* d_scores,
    int total_rays, int max_hits, int k_eids, int k_cent, int ntok, int num_centroids,
    const int64_t* d_csr_row_ptrs, const void* d_csr_col_eids, int eid_bytes,
    const int64_t* d_csr_block_sums, const uint16_t* d_csr_lengths,
    const int* d_prim_eid, int E,
    int buf_idx, cudaStream_t stream
) {
    ensure_task_bufs(buf_idx);
    Bufs& B = g_b[buf_idx];
    int active_ntok = (ntok <= 32) ? ntok : 32;
    cudaMemsetAsync(B.num_tasks, 0, sizeof(int), stream);
    cudaMemsetAsync(B.queue_counter, 0, sizeof(int), stream);
    cudaMemsetAsync(B.num_dropped, 0, sizeof(int), stream);
    cudaMemsetAsync(B.total_postings, 0, sizeof(unsigned long long), stream);

    int threads = WARPS_PER_BLOCK * 32;
    int blocks = (total_rays * 32 + threads - 1) / threads;
    if (eid_bytes == 1) {
        collect_posting_tasks_csr_warp_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
            d_out_c, d_out_v, d_out_hit_count, d_topc, d_scores,
            d_csr_row_ptrs, (const uint8_t*)d_csr_col_eids, d_csr_block_sums, d_csr_lengths,
            total_rays, max_hits, k_eids, k_cent, active_ntok, num_centroids,
            B.num_tasks, B.num_dropped, B.task_qt, B.task_offset, B.task_length, B.task_base,
            B.task_start, B.total_postings, d_prim_eid, E, g_cw_weight, g_compact_scores, stats_ptr());
    } else {
        collect_posting_tasks_csr_warp_kernel<uint16_t><<<blocks, threads, 0, stream>>>(
            d_out_c, d_out_v, d_out_hit_count, d_topc, d_scores,
            d_csr_row_ptrs, (const uint16_t*)d_csr_col_eids, d_csr_block_sums, d_csr_lengths,
            total_rays, max_hits, k_eids, k_cent, active_ntok, num_centroids,
            B.num_tasks, B.num_dropped, B.task_qt, B.task_offset, B.task_length, B.task_base,
            B.task_start, B.total_postings, d_prim_eid, E, g_cw_weight, g_compact_scores, stats_ptr());
    }
    if (stats_ptr()) add_postings_stat_kernel<<<1, 1, 0, stream>>>(B.total_postings, stats_ptr());
}

// Dense stage 3: writes d_passage_scores[N]; the caller takes the top-k over all N.
extern "C" void ragrt_stage3_dense(
    const uint8_t* d_map_packed_24, float* d_passage_scores, int num_passages,
    const uint32_t* d_doc_predicates, uint32_t query_mask, int ntok,
    int buf_idx, cudaStream_t stream
) {
    ensure_dense(buf_idx, num_passages);
    Bufs& B = g_b[buf_idx];
    int active_ntok = (ntok <= 32) ? ntok : 32;
    cudaMemsetAsync(B.qt_pid_max, 0, (size_t)active_ntok * num_passages * sizeof(half), stream);
    int threads = 256;
    cooperative_score_passages_csr_fp16_filtered_kernel<<<142, threads, 0, stream>>>(
        B.num_tasks, B.queue_counter, B.task_qt, B.task_offset, B.task_length, B.task_base,
        d_map_packed_24, num_passages, B.qt_pid_max, d_doc_predicates, query_mask);
    sum_passages_fp16_to_fp32_kernel<<<(num_passages + threads - 1) / threads, threads, 0, stream>>>(
        B.qt_pid_max, active_ntok, num_passages, d_passage_scores, d_doc_predicates, query_mask);
}

// Sparse stage 3. Two small host syncs (number of postings P, then number of touched
// passages U). Produces L = max(U, min_len) entries (pid, score) in *d_pids / *d_sums: the
// touched passages that pass the filter, then (-1, -inf) padding. *n_valid = U.
// The caller takes the top-k over those L entries (none needed when U <= k). Returns L.
extern "C" long long ragrt_stage3_sparse(
    const uint8_t* d_map_packed_24, int num_passages,
    const uint32_t* d_doc_predicates, uint32_t query_mask, long long min_len,
    int buf_idx, cudaStream_t stream, int** d_pids, float** d_sums, long long* n_valid
) {
    Bufs& B = g_b[buf_idx];
    cudaMemcpyAsync(B.h_total, B.total_postings, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    long long P = (long long)(*B.h_total);
    if (g_stage3_mode == 2) {
        ensure_sparse(buf_idx, (P > min_len ? P : min_len) + 1, stream);
        int pid_bits = 1;
        while ((1LL << pid_bits) <= (long long)num_passages) ++pid_bits;
        int end_bit = 5 + pid_bits;          // the sentinel's low end_bit bits are all ones: it sorts last
        long long capk = B.sp_cap;
        uint32_t* ka = reinterpret_cast<uint32_t*>(B.keys_a);
        uint16_t* va = reinterpret_cast<uint16_t*>(ka + capk);
        uint32_t* kb = reinterpret_cast<uint32_t*>(B.keys_b);
        uint16_t* vb = reinterpret_cast<uint16_t*>(kb + capk);
        uint32_t* uk = reinterpret_cast<uint32_t*>(B.run_pid_out);
        uint16_t* um = reinterpret_cast<uint16_t*>(B.sums);
        int threads = 256;
        long long U = 0;
        if (P > 0) {
            sparse_expand32_kernel<<<1024, threads, 0, stream>>>(
                B.num_tasks, B.task_qt, B.task_offset, B.task_length, B.task_base, B.task_start, P,
                d_map_packed_24, num_passages, d_doc_predicates, query_mask, ka, va,
                g_tok_ids, g_res, (g_tok_ids && g_res) ? g_res_lut : nullptr);
            size_t tb = B.temp_bytes;
            rcub::DeviceRadixSort::SortPairs(B.temp, tb, ka, kb, va, vb, (int)P, 0, end_bit, stream);
            tb = B.temp_bytes;
            rcub::DeviceReduce::ReduceByKey(B.temp, tb, kb, uk, vb, um, B.num_runs1, MaxOpU16(), (int)P, stream);
            pairs_to_pid_kernel<<<(unsigned)((P + threads - 1) / threads), threads, 0, stream>>>(
                uk, um, B.num_runs1, P, B.run_pid_in, B.val_in);
            tb = B.temp_bytes;
            rcub::DeviceReduce::ReduceByKey(B.temp, tb, B.run_pid_in, B.run_pid_out, B.val_in, B.sums,
                                            B.num_runs, SumOp(), (int)P, stream);
            cudaMemcpyAsync(B.h_total, B.num_runs, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            U = *reinterpret_cast<int*>(B.h_total);
        } else {
            cudaMemsetAsync(B.num_runs, 0, sizeof(int), stream);
        }
        long long L = U > min_len ? U : min_len;
        sparse_fixup_kernel<<<(unsigned)((L + threads - 1) / threads), threads, 0, stream>>>(
            B.num_runs, L, num_passages, B.run_pid_out, B.sums);
        *d_pids = B.run_pid_out;
        *d_sums = B.sums;
        *n_valid = U;
        return L;
    }
    if (g_stage3_mode == 1) {
        // tables of 2^k >= 2P slots (load <= 0.5); table 1 in keys_a (8 B/slot), table 2 in run_pid_in / val_in
        int log2cap = 16;
        while ((1LL << log2cap) < 2 * P) ++log2cap;
        long long cap = 1LL << log2cap;
        ensure_sparse(buf_idx, (cap > min_len ? cap : min_len) + 1, stream);
        cudaMemsetAsync(B.keys_a, 0, cap * sizeof(unsigned long long), stream);
        cudaMemsetAsync(B.run_pid_in, 0, cap * sizeof(int), stream);
        cudaMemsetAsync(B.val_in, 0, cap * sizeof(float), stream);
        cudaMemsetAsync(B.num_runs, 0, sizeof(int), stream);
        int threads = 256;
        if (P > 0) {
            hash_insert_kernel<<<1024, threads, 0, stream>>>(
                B.num_tasks, B.task_qt, B.task_offset, B.task_length, B.task_base,
                d_map_packed_24, num_passages, d_doc_predicates, query_mask, B.keys_a, log2cap,
                g_tok_ids, g_res, (g_tok_ids && g_res) ? g_res_lut : nullptr);
            unsigned nb = (unsigned)((cap + threads - 1) / threads); if (nb > 4096) nb = 4096;
            hash_reduce_kernel<<<nb, threads, 0, stream>>>(B.keys_a, cap, log2cap, B.run_pid_in, B.val_in);
            hash_compact_kernel<<<nb, threads, 0, stream>>>(B.run_pid_in, B.val_in, cap, B.num_runs, B.run_pid_out, B.sums);
        }
        cudaMemcpyAsync(B.h_total, B.num_runs, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        long long U = *reinterpret_cast<int*>(B.h_total);
        long long L = U > min_len ? U : min_len;
        sparse_fixup_kernel<<<(unsigned)((L + threads - 1) / threads), threads, 0, stream>>>(
            B.num_runs, L, num_passages, B.run_pid_out, B.sums);
        *d_pids = B.run_pid_out;
        *d_sums = B.sums;
        *n_valid = U;
        return L;
    }
    ensure_sparse(buf_idx, (P > min_len ? P : min_len) + 1, stream);

    int pid_bits = 1;
    while ((1LL << pid_bits) <= (long long)num_passages) ++pid_bits;   // 2^bits - 1 >= N: sentinel pid is invalid
    int end_bit = SP_PID_SHIFT + pid_bits;
    unsigned long long sentinel = (end_bit >= 64) ? ~0ULL : ((1ULL << end_bit) - 1ULL);

    int threads = 256;
    long long U = 0;
    if (P > 0) {
        sparse_expand_kernel<<<1024, threads, 0, stream>>>(
            B.num_tasks, B.task_qt, B.task_offset, B.task_length, B.task_base, B.task_start, P,
            d_map_packed_24, num_passages, d_doc_predicates, query_mask, sentinel, B.keys_a,
            g_tok_ids, g_res, (g_tok_ids && g_res) ? g_res_lut : nullptr, nullptr);
        size_t tb = B.temp_bytes;
        rcub::DeviceRadixSort::SortKeys(B.temp, tb, B.keys_a, B.keys_b, (int)P, 0, end_bit, stream);
        sparse_tail_values_kernel<<<(unsigned)((P + threads - 1) / threads), threads, 0, stream>>>(
            B.keys_b, P, num_passages, B.run_pid_in, B.val_in);
        tb = B.temp_bytes;
        rcub::DeviceReduce::ReduceByKey(B.temp, tb, B.run_pid_in, B.run_pid_out, B.val_in, B.sums,
                                        B.num_runs, SumOp(), (int)P, stream);
        int h_runs = 0;
        cudaMemcpyAsync(B.h_total, B.num_runs, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_runs = *reinterpret_cast<int*>(B.h_total);
        U = h_runs;
    } else {
        cudaMemsetAsync(B.num_runs, 0, sizeof(int), stream);
    }
    long long L = U > min_len ? U : min_len;
    // The last run may be the sentinel (filtered / invalid postings): fixup turns it into padding.
    sparse_fixup_kernel<<<(unsigned)((L + threads - 1) / threads), threads, 0, stream>>>(
        B.num_runs, L, num_passages, B.run_pid_out, B.sums);
    *d_pids = B.run_pid_out;
    *d_sums = B.sums;
    *n_valid = U;
    return L;
}

// Sparse stage 3 with NO host synchronization (for CUDA graphs): sorts a fixed `cap` keys, the
// P real ones (P = postings counted by Stage 2, on the device) followed by sentinels. The caller
// must have reserved cap (ragrt_reserve_sparse) and read P beforehand to pick cap >= P + 1;
// postings beyond cap would be dropped. Produces cap entries (pid, score) in *d_pids / *d_sums:
// the touched passages, then (-1, -inf) padding. The filter mask is read from d_mask.
extern "C" long long ragrt_stage3_sparse_static(
    const uint8_t* d_map_packed_24, int num_passages, const uint32_t* d_doc_predicates,
    const uint32_t* d_mask, long long cap, int buf_idx, cudaStream_t stream, int** d_pids, float** d_sums
) {
    Bufs& B = g_b[buf_idx];
    if (!B.init || cap > B.sp_cap || cap <= 0) return -1;
    int pid_bits = 1;
    while ((1LL << pid_bits) <= (long long)num_passages) ++pid_bits;
    int end_bit = SP_PID_SHIFT + pid_bits;
    unsigned long long sentinel = (end_bit >= 64) ? ~0ULL : ((1ULL << end_bit) - 1ULL);
    int threads = 256;
    sparse_fill_sentinel_kernel<<<256, threads, 0, stream>>>(B.total_postings, cap, sentinel, B.keys_a);
    sparse_expand_kernel<<<1024, threads, 0, stream>>>(
        B.num_tasks, B.task_qt, B.task_offset, B.task_length, B.task_base, B.task_start, cap,
        d_map_packed_24, num_passages, d_doc_predicates, 0u, sentinel, B.keys_a,
        g_tok_ids, g_res, (g_tok_ids && g_res) ? g_res_lut : nullptr, d_mask);
    size_t tb = B.temp_bytes;
    rcub::DeviceRadixSort::SortKeys(B.temp, tb, B.keys_a, B.keys_b, (int)cap, 0, end_bit, stream);
    sparse_tail_values_kernel<<<(unsigned)((cap + threads - 1) / threads), threads, 0, stream>>>(
        B.keys_b, cap, num_passages, B.run_pid_in, B.val_in);
    tb = B.temp_bytes;
    rcub::DeviceReduce::ReduceByKey(B.temp, tb, B.run_pid_in, B.run_pid_out, B.val_in, B.sums,
                                    B.num_runs, SumOp(), (int)cap, stream);
    sparse_fixup_kernel<<<(unsigned)((cap + threads - 1) / threads), threads, 0, stream>>>(
        B.num_runs, cap, num_passages, B.run_pid_out, B.sums);
    *d_pids = B.run_pid_out;
    *d_sums = B.sums;
    return cap;
}

// Allocate the task and sparse buffers up front (outside any graph capture). Returns the
// capacity actually available (>= cap). Growing it frees the old buffers, so CUDA graphs
// captured against them must be discarded (the Python side compares sparse_capacity()).
extern "C" long long ragrt_reserve_sparse(long long cap, int buf_idx, cudaStream_t stream) {
    ensure_task_bufs(buf_idx);
    ensure_sparse(buf_idx, cap, stream);
    return g_b[buf_idx].sp_cap;
}
extern "C" long long ragrt_sparse_capacity(int buf_idx) { return g_b[buf_idx].init ? g_b[buf_idx].sp_cap : 0; }
extern "C" unsigned long long* ragrt_total_postings_ptr(int buf_idx) {
    ensure_task_bufs(buf_idx);
    return g_b[buf_idx].total_postings;
}

extern "C" long long ragrt_last_total_postings(int buf_idx) {
    Bufs& B = g_b[buf_idx];
    if (!B.init) return 0;
    unsigned long long h = 0;
    cudaMemcpy(&h, B.total_postings, sizeof(h), cudaMemcpyDeviceToHost);
    return (long long)h;
}

extern "C" int ragrt_dropped_task_count(int buf_idx) {
    int h = 0;
    if (buf_idx < 0 || buf_idx > 1 || !g_b[buf_idx].init) return 0;
    cudaMemcpy(&h, g_b[buf_idx].num_dropped, sizeof(int), cudaMemcpyDeviceToHost);
    return h;
}
