"""
Sanity-check a built RAGRT index directory (no GPU, no ColBERT needed).

  python3 check_index.py <sparse_csr_dir> [<sparse_csr_dir> ...]

Checks: CSR shape invariants, block sums, codebook/rotation fingerprint, eid width vs E,
pid range (max pid vs passage count, share of pids >= 2^22 that the old
int32 packing used to corrupt). Reads csr_packed_24.npy in chunks via mmap.
"""
import os, sys, json
import numpy as np

import ragrt_index_lib as L

CHUNK_POSTINGS = 100_000_000


def check(d):
    print(f"\n== {d}")
    ok = True

    def expect(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    row_ptrs = np.load(os.path.join(d, "csr_row_ptrs.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(d, "csr_lengths.npy"), mmap_mode="r")
    eids = np.load(os.path.join(d, "csr_col_eids.npy"), mmap_mode="r")
    bsums = np.load(os.path.join(d, "csr_block_sums_128.npy"))
    packed = np.load(os.path.join(d, "csr_packed_24.npy"), mmap_mode="r")
    cb = np.load(os.path.join(d, "codebooks.npy"))
    meta = json.load(open(os.path.join(d, "codebook_meta.json")))
    R = np.load(os.path.join(d, "svd_rotation_128_to_96.npy"))
    pred_file = next((f for f in ("predicates.npy", "synthetic_predicates.npy")
                      if os.path.exists(os.path.join(d, f))), None)
    assert pred_file, "no predicates.npy in the index directory (run build_predicates.py)"
    num_passages = len(np.load(os.path.join(d, pred_file), mmap_mode="r"))

    n_lists = len(lengths)
    n_post = len(packed) // 3
    lsum = int(lengths.sum(dtype=np.int64))
    print(f"  {n_lists:,} lists, {n_post:,} postings, {num_passages:,} passages")

    expect(len(packed) % 3 == 0, "packed length is a multiple of 3")
    expect(lsum == n_post, f"sum(lengths) == postings ({lsum:,})")
    expect(len(eids) == n_lists, "one eid per list")
    expect(row_ptrs.shape[0] == L.NUM_SUBSPACES, f"row_ptrs has {L.NUM_SUBSPACES} subspace rows")
    expect(int(row_ptrs[0, 0]) == 0 and int(row_ptrs[-1, -1]) == n_lists, "row_ptrs span [0, num_lists]")
    expect(bool(np.all(np.diff(np.asarray(row_ptrs).ravel().astype(np.int64)) >= 0)), "row_ptrs non-decreasing")
    expect(int(lengths.min()) >= 1, "no empty lists")
    expect(np.array_equal(bsums, L.block_sums(np.asarray(lengths))), "block sums match lengths")

    E = cb.shape[1]
    expect(cb.shape[0] == L.NUM_SUBSPACES and cb.shape[2] == L.SUBSPACE_DIM and E <= L.MAX_EIDS,
           f"codebooks shape {tuple(cb.shape)}")
    expect(eids.dtype == L.eid_dtype(E), f"csr_col_eids dtype {eids.dtype} matches E={E}")
    mx = 0
    for a in range(0, n_lists, CHUNK_POSTINGS):
        mx = max(mx, int(np.asarray(eids[a:a + CHUNK_POSTINGS]).max()))
    expect(mx < E, f"all eids < E (max {mx})")
    expect(row_ptrs.dtype in (np.int32, np.int64), f"row_ptrs dtype {row_ptrs.dtype}")
    expect(meta.get("rotation_sha256") == L.rotation_fingerprint(R), "codebooks match this rotation")
    print(f"  codebook mean rel_err {meta.get('mean_rel_err', float('nan')):.3f}")

    max_pid, hi_count = -1, 0
    for a in range(0, n_post, CHUNK_POSTINGS):
        pids = L.unpack_pids_24(packed[3 * a: 3 * min(a + CHUNK_POSTINGS, n_post)])
        max_pid = max(max_pid, int(pids.max()))
        hi_count += int(np.count_nonzero(pids >= (1 << 22)))
    expect(max_pid == num_passages - 1, f"max pid {max_pid:,} == num_passages - 1")
    print(f"  postings with pid >= 2^22 (old bug range): {hi_count:,} ({hi_count / n_post:.1%})")
    tp = os.path.join(d, "csr_token_ids.npy")
    if os.path.exists(tp):          # token-level plain IVF (residual ablation)
        tids = np.load(tp, mmap_mode="r")
        expect(len(tids) == n_post and tids.dtype == np.uint32, f"one uint32 token id per posting ({len(tids):,})")
        mono, tmax = True, 0
        for a in range(0, n_post, CHUNK_POSTINGS):
            t = np.asarray(tids[a:a + CHUNK_POSTINGS]).astype(np.int64)
            tmax = max(tmax, int(t.max()))
            # tokens are numbered in passage order, so pid must be non-decreasing in token id
            p = L.unpack_pids_24(packed[3 * a: 3 * min(a + CHUNK_POSTINGS, n_post)])
            o = np.argsort(t, kind="stable")
            mono &= bool(np.all(np.diff(p[o]) >= 0))
        expect(mono, "pid is non-decreasing in token id (token ids match pids)")
        expect(tmax + 1 == n_post, f"token ids cover 0..{n_post - 1:,} (max {tmax:,})")
    return ok


if __name__ == "__main__":
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    results = [check(d) for d in dirs]
    print("\nALL OK" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)
