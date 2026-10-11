"""
Build MUVERA document FDEs from the exported ColBERT doc vectors (export_embeddings.py), main env.

  python3 build_muvera_fde.py --emb_dir /local/scratch/a/cashman3/lotte_doc_embs \
      --outdir /local/scratch/a/cashman3/muvera_lotte_r20_k4_p8 --reps 20 --k_sim 4 --d_proj 8

Writes <outdir>/fde_docs_fp16.npy [passages, reps * 2^k_sim * d_proj], fde_config.json,
fde_params.pt. Size: passages x fde_dim x 2 bytes (LoTTE, 2560 dims: 12.4 GB); the whole
matrix is loaded onto the GPU at search time. Then run_benchmarks.py --engines muvera
--muvera_dir <outdir>.
"""
import os, sys, shutil, argparse, time
import numpy as np
import torch
import muvera_fde as M

TOKENS_PER_CHUNK = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--k_sim", type=int, default=4)
    ap.add_argument("--d_proj", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    embs = np.load(os.path.join(args.emb_dir, "doc_embs_fp16.npy"), mmap_mode="r")
    offs = np.load(os.path.join(args.emb_dir, "doc_offsets.npy"))
    N = len(offs) - 1
    assert (np.diff(offs) > 0).all(), "every passage needs at least one token"
    cfg, params = M.make_params(embs.shape[1], args.reps, args.k_sim, args.d_proj, args.seed)
    need = N * cfg["fde_dim"] * 2
    os.makedirs(args.outdir, exist_ok=True)
    free = shutil.disk_usage(args.outdir).free
    if free < need + 2 * 2 ** 30:
        sys.exit(f"FATAL: need {need / 2**30:.1f} GiB + 2 GiB free in {args.outdir}, have {free / 2**30:.1f} GiB")
    print(f"{N:,} passages, FDE dim {cfg['fde_dim']} ({need / 2**30:.1f} GiB)", flush=True)
    M.save_params(args.outdir, cfg, params)
    params = {k: v.cuda() for k, v in params.items()}
    out = np.lib.format.open_memmap(os.path.join(args.outdir, "fde_docs_fp16.npy"), mode="w+",
                                    dtype=np.float16, shape=(N, cfg["fde_dim"]))
    t0, a = time.time(), 0
    while a < N:
        b = int(np.searchsorted(offs, offs[a] + TOKENS_PER_CHUNK, side="right")) - 1
        b = max(b, a + 1)
        b = min(b, N)
        X = torch.from_numpy(np.asarray(embs[offs[a]:offs[b]])).cuda().float()
        lens = torch.from_numpy(np.diff(offs[a:b + 1])).cuda()
        doc = torch.repeat_interleave(torch.arange(b - a, device="cuda"), lens)
        out[a:b] = M.doc_fdes(X, doc, b - a, params, cfg).half().cpu().numpy()
        if (a // 100_000) != (b // 100_000) or b == N:
            print(f"  {b:,} / {N:,} passages ({time.time() - t0:.0f} s)", flush=True)
        a = b
    out.flush()
    print(f"Wrote {args.outdir}/fde_docs_fp16.npy")


if __name__ == "__main__":
    main()
