"""
Inputs for juno_bench.cu: a JUNO-style IVFPQ index (JUNO Alg. 1: IVF clusters, then a codebook of E
entries per 2D subspace trained on the residuals) and, for every query, its residual projections for
the max_nprobe nearest clusters, i.e. the ray origins of JUNO Alg. 2.

Thresholds: JUNO picks a per-query-projection radius with a density regressor; the RT-vs-CUDA
question does not depend on how the radius is chosen, only on how many entries it selects. So each
threshold here is set per subspace to select a target fraction of the E entries on average
(--fracs; JUNO reports 25 to 30% of entries used per subspace, Sec. 3.2).

Coordinates are scaled so the largest radius is 0.9 R (spheres of radius R must contain every
threshold, and spheres of neighbouring subspaces, 2 apart in z, must not overlap; R < 1).

  python3 prepare_juno_data.py --data sift1m --sift_dir /path/to/sift --out DIR
  python3 prepare_juno_data.py --data gaussian --out DIR      (no download needed)
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from gpu_kmeans import kmeans, assign_np


def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def gaussian_data(n, nq, d, seed=0):
    """Clustered synthetic data (1,000 Gaussian blobs) so IVF and PQ behave like on real embeddings."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (1000, d)).astype(np.float32) * 4
    lab = rng.integers(0, 1000, n + nq)
    x = centers[lab] + rng.normal(0, 1, (n + nq, d)).astype(np.float32)
    return x[:n], x[n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["sift1m", "gaussian"], default="sift1m")
    ap.add_argument("--sift_dir", default=None, help="directory with sift_base.fvecs and sift_query.fvecs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nq", type=int, default=10000)
    ap.add_argument("--n", type=int, default=1_000_000, help="base points (gaussian data only)")
    ap.add_argument("--nlist", type=int, default=4096, help="IVF clusters (JUNO: IVF4096 on DEEP1M)")
    ap.add_argument("--max_nprobe", type=int, default=64)
    ap.add_argument("--E", type=int, default=256, help="codebook entries per 2D subspace (8-bit PQ)")
    ap.add_argument("--fracs", default="0.05,0.1,0.25,0.5")
    ap.add_argument("--R", type=float, default=0.95)
    ap.add_argument("--train", type=int, default=200_000, help="residuals used to train each subspace codebook")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.data == "sift1m":
        base = read_fvecs(os.path.join(a.sift_dir, "sift_base.fvecs"))
        queries = read_fvecs(os.path.join(a.sift_dir, "sift_query.fvecs"))[:a.nq]
    else:
        base, queries = gaussian_data(a.n, a.nq, 128)
    n, D = base.shape
    S = D // 2
    print(f"{a.data}: {n:,} x {D}, {len(queries):,} queries, nlist {a.nlist}, S={S} subspaces x E={a.E}", flush=True)

    cent = kmeans(base, a.nlist, iters=20)
    lab, _ = assign_np(base, cent)
    resid = base - cent[lab]
    rng = np.random.default_rng(0)
    tr = resid[rng.choice(n, min(a.train, n), replace=False)]
    codebook = np.zeros((S, a.E, 2), np.float32)
    for s in range(S):
        codebook[s] = kmeans(tr[:, 2 * s:2 * s + 2], a.E, iters=20, init_sample=min(len(tr), 50_000))
    print("  IVF and subspace codebooks trained", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Qt, Ct = torch.as_tensor(queries, device=dev), torch.as_tensor(cent, device=dev)
    d2 = (Qt * Qt).sum(1, keepdim=True) - 2 * Qt @ Ct.T + (Ct * Ct).sum(1)[None, :]
    probe = d2.topk(a.max_nprobe, dim=1, largest=False).indices                 # [Q, P], nearest first
    origins = (Qt[:, None, :] - Ct[probe]).reshape(len(queries), a.max_nprobe, S, 2).float().cpu().numpy()

    fracs = [float(f) for f in a.fracs.split(",")]
    nr = len(queries) * a.max_nprobe
    sample = origins.reshape(-1, S, 2)[rng.choice(nr, min(4096, nr), replace=False)]                    # [<=4096, S, 2]
    dist = np.sqrt(((sample[:, :, None, :] - codebook[None, :, :, :]) ** 2).sum(-1))                   # [4096, S, E]
    thr = np.stack([np.quantile(dist.transpose(1, 0, 2).reshape(S, -1), f, axis=1) for f in fracs]).astype(np.float32)
    scale = 0.9 * a.R / float(thr.max())
    codebook *= scale; origins *= scale; thr *= scale

    origins.astype(np.float32).tofile(os.path.join(a.out, "origins.bin"))
    codebook.astype(np.float32).tofile(os.path.join(a.out, "codebook.bin"))
    thr.astype(np.float32).tofile(os.path.join(a.out, "thresholds.bin"))
    with open(os.path.join(a.out, "fracs.txt"), "w") as f:
        f.write("\n".join(str(x) for x in fracs) + "\n")
    with open(os.path.join(a.out, "meta.txt"), "w") as f:
        f.write(f"{S} {a.E} {len(queries)} {a.max_nprobe} {a.R}\n")
    print(f"  wrote {a.out}: origins {origins.shape}, thresholds {thr.shape} (scale {scale:.4g})", flush=True)


if __name__ == "__main__":
    main()
