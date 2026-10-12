"""
Plain IVF index in RAGRT's CSR format, for the ragrt_ivf ablation engine (engines.py).

One posting list per ColBERT centroid with the passages that have a token assigned to it
(deduplicated per (centroid, passage), like ColBERT's own IVF). Everything lives in
"subspace" 0; subspaces 1-31 are empty. Lists longer than the uint16 length field are split
across eids 0, 1, 2, ... of the same centroid. ragrt_ivf runs Stage 1 in all-codewords mode
with codeword weight 0, so every chunk of every probed centroid is fetched and scored q.c:
PLAID's candidate generation (centroid interaction, no pruning threshold) on RAGRT's
Stage 2/3 kernels and the same TileMaxSim rerank.

Rotation, 96D centroids, codebook metadata and predicates come from an existing RAGRT index
(--src_dir) so the query side is identical; codebooks.npy keeps the first E_ivf codewords of
the source (their values are unused: weight 0).

  python3 build_plain_ivf.py --index <colbert index> --collection <collection.tsv> \
      --src_dir /local/scratch/a/cashman3/juno_pq_lotte_full_sparse8 \
      --outdir /local/scratch/a/cashman3/lotte_plain_ivf
"""
import os, sys, json, shutil, argparse
import numpy as np

import ragrt_index_lib as L

COPY = ["svd_rotation_128_to_96.npy", "centroids_96d_svd.npy", "codebook_meta.json", "predicates.npy"]


def plain_ivf_csr(codes, pids, num_centroids, max_len=L.MAX_LIST_LENGTH, dedup=True, tokens=False):
    """
    codes, pids: per token (token order = the order of ColBERT's codes / residuals). Returns dict
    with row_ptrs int64 [32, num_centroids + 1], col_eids, lengths (int64), pids_sorted, E_needed
    (chunks of the longest centroid). tokens=True: one posting per TOKEN (no dedup), sorted by
    (centroid, token id), plus token_ids (uint32) aligned with pids_sorted, for residual scoring.
    """
    codes = np.asarray(codes, dtype=np.int64)
    pids = np.asarray(pids, dtype=np.int64)
    assert len(codes) == len(pids)
    assert pids.min() >= 0 and pids.max() <= L.MAX_PID
    tids = None
    if tokens:
        assert len(codes) < (1 << 32)
        key = np.sort((codes << 32) | np.arange(len(codes), dtype=np.int64))
        c, tids = key >> 32, (key & 0xFFFFFFFF)
        p = pids[tids]
        tids = tids.astype(np.uint32)
    else:
        key = (codes << 24) | pids
        key = np.unique(key) if dedup else np.sort(key)
        c, p = key >> 24, key & ((1 << 24) - 1)
    per_c = np.bincount(c, minlength=num_centroids).astype(np.int64)
    nchunk = (per_c + max_len - 1) // max_len                     # 0 for empty centroids
    n_lists = int(nchunk.sum())
    lengths = np.empty(n_lists, dtype=np.int64)
    eids = np.empty(n_lists, dtype=np.int64)
    pos = 0
    for cid in np.flatnonzero(nchunk):
        k, n = int(nchunk[cid]), int(per_c[cid])
        lengths[pos:pos + k] = max_len
        lengths[pos + k - 1] = n - max_len * (k - 1)
        eids[pos:pos + k] = np.arange(k)
        pos += k
    row = np.zeros(num_centroids + 1, dtype=np.int64)
    row[1:] = np.cumsum(nchunk)
    row_ptrs = np.empty((L.NUM_SUBSPACES, num_centroids + 1), dtype=np.int64)
    row_ptrs[0] = row
    row_ptrs[1:] = n_lists                                        # empty subspaces
    return {"row_ptrs": row_ptrs, "col_eids": eids, "lengths": lengths, "pids_sorted": p,
            "token_ids": tids, "E_needed": int(nchunk.max()) if n_lists else 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--src_dir", required=True, help="an existing RAGRT index of the same ColBERT index")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tokens", action="store_true",
                    help="one posting per token + csr_token_ids.npy (for residual scoring); default dedups per passage")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../reference/colbert-plaid"))
    from colbert import Searcher
    from colbert.indexing.loaders import load_doclens
    from colbert.utils.utils import flatten

    searcher = Searcher(index=args.index, collection=args.collection)
    num_centroids = searcher.ranker.codec.centroids.shape[0]
    doclens = np.asarray(flatten(load_doclens(args.index, flatten=False)), dtype=np.int64)
    codes = searcher.ranker.embeddings.codes.numpy()
    total = L.num_real_tokens(codes.shape[0], doclens.sum())
    codes = codes[:total]
    pids = np.repeat(np.arange(len(doclens), dtype=np.int64), doclens)
    print(f"{len(doclens):,} passages, {total:,} tokens, {num_centroids:,} centroids", flush=True)
    del searcher

    csr = plain_ivf_csr(codes, pids, num_centroids, tokens=args.tokens)
    E_src = np.load(os.path.join(args.src_dir, "codebooks.npy"), mmap_mode="r").shape[1]
    E = max(4, csr["E_needed"])
    if E > min(E_src, L.MAX_EIDS_UINT8):
        sys.exit(f"FATAL: longest centroid needs {E} chunks; the source codebook has E={E_src}")
    n_post = len(csr["pids_sorted"])
    print(f"{len(csr['lengths']):,} lists, {n_post:,} postings{'' if args.tokens else ' after dedup'} ({n_post / total:.2f} per token), "
          f"E={E}", flush=True)

    lengths = csr["lengths"].astype(np.uint16)
    bsums = L.block_sums(lengths)
    probe = np.unique(np.concatenate([[0, len(lengths) - 1],
                                      np.random.default_rng(0).integers(0, len(lengths), 8000)]))
    want = np.concatenate([[0], np.cumsum(csr["lengths"])[:-1]])[probe]
    assert np.array_equal(L.offsets_from_block_sums(bsums, lengths, probe), want), "block-sum offsets"
    packed = L.pack_pids_24(csr["pids_sorted"])
    assert np.array_equal(L.unpack_pids_24(packed[:3 * 100000]), csr["pids_sorted"][:100000])

    np.save(os.path.join(args.outdir, "csr_row_ptrs.npy"), csr["row_ptrs"])
    np.save(os.path.join(args.outdir, "csr_col_eids.npy"), csr["col_eids"].astype(np.uint8))
    np.save(os.path.join(args.outdir, "csr_lengths.npy"), lengths)
    np.save(os.path.join(args.outdir, "csr_block_sums_128.npy"), bsums)
    np.save(os.path.join(args.outdir, "csr_packed_24.npy"), packed)
    if args.tokens:
        np.save(os.path.join(args.outdir, "csr_token_ids.npy"), csr["token_ids"])
    np.save(os.path.join(args.outdir, "codebooks.npy"),
            np.ascontiguousarray(np.load(os.path.join(args.src_dir, "codebooks.npy"))[:, :E]))
    for f in COPY:
        shutil.copy(os.path.join(args.src_dir, f), os.path.join(args.outdir, f))
    json.dump({"kind": "plain_ivf", "dedup": not args.tokens, "tokens": bool(args.tokens), "E": E, "lists": len(lengths), "postings": n_post,
               "src_dir": args.src_dir}, open(os.path.join(args.outdir, "plain_ivf_meta.json"), "w"), indent=1)
    print(f"[DONE] {args.outdir} ({3 * n_post / 1e9:.2f} GB packed). Check: python3 check_index.py {args.outdir}")


if __name__ == "__main__":
    main()
