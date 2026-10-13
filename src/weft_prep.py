"""
Query prep (Stage 0b) for the plain-IVF engine, pure torch so it is unit-testable on CPU.

  native_prep(Q, C, nc, impute)
    Q: [ntok, 128] query token embeddings (fp16 on the GPU), C: [num_centroids, 128] ColBERT
    centroids (fp16). Centroid scores come from one fp16 tensor-core GEMM in the native 128
    dimensions (PLAID's own centroid scores), not the 96D SVD projection the codebook engines need.

    impute=True folds WARP-style missing-score imputation into the scores. For query token t,
    every centroid that was not probed scores at most m_t = the (nc+1)-th best centroid score, so
    a passage that token t did not reach is credited m_t instead of 0. Because
        sum_t [found ? best_t : m_t] = sum_t m_t + sum_{t found} (best_t - m_t)
    and sum_t m_t is the same for every passage, ranking by the imputed score is identical to
    ranking by sum over found tokens of (best_t - m_t). So the scores handed to Stage 2 are
    shifted by -m_t per token; no kernel change. (Stage 2 keeps only positive base scores; a
    probed centroid's shifted score is >= 0, with equality only on exact ties.)

  Returns (topc int32 [ntok, nc], scores float32 [ntok, num_centroids], m float32 [ntok] or None).
  compact=True returns scores [ntok, nc] aligned with topc instead (Stage 2 reads them with
  set_compact_scores(True)); it skips building the full float32 matrix.

  residual_lut(Q, bucket_weights, reversed_bit_map, decomp_table)
    WARP-style implicit decompression for the token-level index: a 2-bit ColBERT residual is
    32 bytes, byte j holding dims 4j..4j+3 (through reversed_bit_map and decomp_table, exactly as
    the rerank kernel decodes it). lut[t][j][b] = sum_k Q[t][4j+k] * w[decomp[rbm[b]][k]], so
    q_t . residual(token) = sum_j lut[t][j][byte_j]: 32 lookups per posting instead of 128 MACs.
    (No renormalization of centroid + residual, as in WARP.)
"""
import torch


def native_prep(Q, C, nc, impute=False, compact=False):
    s = Q @ C.T                                   # fp16 in, fp16 out (tensor cores on the GPU)
    k = min(nc + (1 if impute else 0), s.shape[1])
    vals, idx = s.topk(k, dim=-1)                 # sorted, best first
    topc = idx[:, :nc].to(torch.int32).contiguous()
    if compact:
        v = vals[:, :nc].float()
        if impute and k > nc:
            m = vals[:, nc].float()
            return topc, (v - m[:, None]).contiguous(), m
        return topc, v.contiguous(), None
    if impute and k > nc:
        m = vals[:, nc].float()
        scores = (s.float() - m[:, None]).contiguous()
        return topc, scores, m
    return topc, s.float().contiguous(), None


def residual_byte_weights(bucket_weights, reversed_bit_map, decomp_table):
    """[256 raw byte, 4 dims]: the bucket weight each byte value decodes to (constant per index)."""
    buckets = decomp_table.long()[reversed_bit_map.long()]             # [256, 4]
    return bucket_weights.float()[buckets].contiguous()


def residual_lut(Q, bucket_weights, reversed_bit_map, decomp_table, wb=None):
    """Q [ntok, 128] -> lut float32 [ntok, 32, 256]. wb: residual_byte_weights(...), cached by the caller."""
    if wb is None:
        wb = residual_byte_weights(bucket_weights, reversed_bit_map, decomp_table)
    Qr = Q.float().view(Q.shape[0], 32, 4)                             # [ntok, 32 bytes, 4 dims]
    return (Qr @ wb.T).contiguous()                                    # one batched GEMM


def residual_dot_reference(q, residual_bytes, bucket_weights, reversed_bit_map, decomp_table):
    """q [128], residual_bytes [32] uint8 -> q . residual, decoded the way the rerank kernel does."""
    buckets = decomp_table.long()[reversed_bit_map.long()[residual_bytes.long()]].reshape(-1)   # [128]
    return float((q.float() * bucket_weights.float()[buckets]).sum())


def imputed_passage_scores(best, found, m):
    """Reference for the tests: best/found [P, T] per passage and token, m [T] -> imputed sums."""
    return torch.where(found, best, m[None, :]).sum(1)


def adaptive_probe(Q, C, nc, ncap, pass_c, tot_c, sel, q=1.0):
    """Adaptive probing under a filter (engine weft_adapt). pass_c / tot_c [NC]: postings per centroid
    whose passage passes the filter / all postings; sel: global pass rate. For each query token the
    target is sel x (postings in its top-nc lists), the passing postings an uncorrelated filter would
    leave; the token needs the smallest nc' in [nc, ncap] whose top-nc' lists hold that many passing
    postings. The query uses the q-quantile of the tokens' nc' (q = 1: the max; one k_cent per query),
    imputation at the (nc'+1)-th score.
    Fast path: when at most a (1 - q) share of tokens is short at nc, nc' = nc and no deeper top-k runs.
    Returns (topc int32 [ntok, nc'], compact shifted scores float32 [ntok, nc'], nc')."""
    s = Q @ C.T
    vals, idx = s.topk(nc + 1, dim=-1)
    target = sel * tot_c[idx[:, :nc]].sum(1)
    short = (pass_c[idx[:, :nc]].sum(1) < target).float().mean()
    if float(short) <= 1.0 - q + 1e-9:
        nc2 = nc
    else:
        vals, idx = s.topk(ncap + 1, dim=-1)
        cum = pass_c[idx[:, :ncap]].cumsum(1)
        need = ((cum < target[:, None]).sum(1) + 1).clamp(nc, ncap).float()
        nc2 = int(torch.quantile(need, q).ceil()) if q < 1.0 else int(need.max())
        nc2 = max(nc, min(ncap, nc2))
    m = vals[:, nc2].float()
    return idx[:, :nc2].to(torch.int32).contiguous(), (vals[:, :nc2].float() - m[:, None]).contiguous(), nc2
