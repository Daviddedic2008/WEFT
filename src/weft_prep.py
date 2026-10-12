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
"""
import torch


def native_prep(Q, C, nc, impute=False):
    s = Q @ C.T                                   # fp16 in, fp16 out (tensor cores on the GPU)
    k = min(nc + (1 if impute else 0), s.shape[1])
    vals, idx = s.topk(k, dim=-1)                 # sorted, best first
    topc = idx[:, :nc].to(torch.int32).contiguous()
    if impute and k > nc:
        m = vals[:, nc].float()
        scores = (s.float() - m[:, None]).contiguous()
        return topc, scores, m
    return topc, s.float().contiguous(), None


def imputed_passage_scores(best, found, m):
    """Reference for the tests: best/found [P, T] per passage and token, m [T] -> imputed sums."""
    return torch.where(found, best, m[None, :]).sum(1)
