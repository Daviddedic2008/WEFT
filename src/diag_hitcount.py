"""
Is an L2 hit-count filter on 3D slices selective enough to be worth ray tracing? (GPU, offline)

Idea: in the rotated 96D frame (unit vectors), ||q - x||^2 = sum over the 32 slices of
||q_s - x_s||^2. A ray per (query token, slice) against small spheres around each point's
slice finds the points within radius r_s of the query in that slice (L2 is local, unlike
inner product, so the BVH can prune). A point "passes" if it is hit in at least H slices.

Two radius schemes are measured:
  scaled : r_s = alpha * RMS slice distance between query tokens and their true top-K
           neighbours (calibrated on half the queries, evaluated on the other half);
           every H from 1 to S is reported.
  bound  : lossless by construction. With eps = the K-th neighbour's squared distance and
           radius^2 = eps / m in every slice, a true neighbour can miss fewer than m of the
           S slices (their squared distances sum to at most eps), so H = S - m + 1 keeps
           recall at 1. eps is the per-query oracle value: the best case for this filter.

Using only the first S slices (S = 1, 2, 3, 4 = 3, 6, 9, 12 dims of the top principal
directions) answers "would a 6/9/12D projection work as a first filter?".

Targets: a random sample of doc tokens (K = 10, 100) and all ColBERT centroids (K = 8, 32).
Reported per (target, K, S): the candidate fraction needed for recall 0.90 / 0.95, and the
RT work it implies (hits per ray, scaled to the full corpus for tokens).

  python3 diag_hitcount.py --dataset lotte [--n_queries 300 --doc_tokens 2000000]
Writes <results>/<dataset>/diag_hitcount.json
"""
import os, json, argparse
import numpy as np
import torch

import eval_config as C
import eval_lib as E
import ragrt_index_lib as L
from engines import Engines
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings

S_LIST = [1, 2, 3, 4, 8, 16, 32]
ALPHAS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
BOUND_M = [1, 2, 4, 8]
NS = L.NUM_SUBSPACES


def slices(x):
    return x.view(x.shape[0], NS, 3)


@torch.no_grad()
def topk_neighbours(Q, X, K, chunk=64):
    out_i, out_v = [], []
    for a in range(0, len(Q), chunk):
        v, i = torch.topk(Q[a:a + chunk] @ X.T, K, dim=1)
        out_i.append(i); out_v.append(v)
    return torch.cat(out_i), torch.cat(out_v)


@torch.no_grad()
def run_target(name, Qcal, Qev, X, K, full_size, chunk):
    N = X.shape[0]
    Xs = slices(X)                                         # [N, 32, 3]
    xsq = (Xs * Xs).sum(-1)                                # [N, 32]

    # radius scale per slice from calibration queries and their true neighbours
    ci, _ = topk_neighbours(Qcal, X, K)
    d2 = ((slices(Qcal)[:, None] - Xs[ci]) ** 2).sum(-1)   # [nc, K, 32]
    rms = d2.mean(dim=(0, 1)).sqrt()                       # [32]

    ei, ev = topk_neighbours(Qev, X, K)
    eps = (2.0 - 2.0 * ev[:, -1]).clamp_min(1e-6)          # K-th neighbour's squared distance (unit vectors)

    nA, nM, Smax = len(ALPHAS), len(BOUND_M), NS
    # histograms over the number of slices hit, at each checkpoint S: [scheme, S, h]
    hist_all = torch.zeros(nA, len(S_LIST), Smax + 1, dtype=torch.float64, device=X.device)
    hist_nb = torch.zeros_like(hist_all)
    hits_ray = torch.zeros(nA, NS, dtype=torch.float64, device=X.device)
    b_pass = torch.zeros(nM, len(S_LIST), dtype=torch.float64, device=X.device)
    b_recall = torch.zeros_like(b_pass)
    b_hits = torch.zeros(nM, NS, dtype=torch.float64, device=X.device)
    nq = len(Qev)
    for a in range(0, nq, chunk):
        q = slices(Qev[a:a + chunk]); b = q.shape[0]
        nb = ei[a:a + chunk]
        cnt = torch.zeros(nA, b, N, dtype=torch.uint8, device=X.device)
        bcnt = torch.zeros(nM, b, N, dtype=torch.uint8, device=X.device)
        e = eps[a:a + chunk]
        for s in range(NS):
            qs = q[:, s]                                   # [b, 3]
            dist2 = (qs * qs).sum(-1, keepdim=True) + xsq[None, :, s] - 2.0 * (qs @ Xs[:, s].T)   # [b, N]
            for j, al in enumerate(ALPHAS):
                hit = dist2 <= float((al * rms[s]) ** 2)
                cnt[j] += hit
                hits_ray[j, s] += hit.sum()
            for j, m in enumerate(BOUND_M):
                hit = dist2 <= (e / m)[:, None]
                bcnt[j] += hit
                b_hits[j, s] += hit.sum()
            if (s + 1) in S_LIST:
                ci_ = S_LIST.index(s + 1)
                for j in range(nA):
                    hist_all[j, ci_] += torch.bincount(cnt[j].view(-1).long(), minlength=Smax + 1)[:Smax + 1].double()
                    hist_nb[j, ci_] += torch.bincount(cnt[j].gather(1, nb).view(-1).long(), minlength=Smax + 1)[:Smax + 1].double()
                for j, m in enumerate(BOUND_M):
                    need = (s + 1) - m + 1      # <= 0 when S < m: the bound keeps everything
                    ok = bcnt[j].int() >= need
                    b_pass[j, ci_] += ok.double().mean(dim=1).sum()
                    b_recall[j, ci_] += ok.gather(1, nb).double().mean(dim=1).sum()
    scale = full_size / N
    res = {"N_sampled": N, "full_size": full_size, "K": K, "queries_eval": nq,
           "rms_slice_dist": rms.tolist(), "scaled": {}, "bound": {}}
    for ci_, S in enumerate(S_LIST):
        res["scaled"][str(S)] = {}
        for j, al in enumerate(ALPHAS):
            tot = hist_all[j, ci_].sum(); totn = hist_nb[j, ci_].sum()
            pts = []
            for H in range(1, S + 1):
                cand = float(hist_all[j, ci_, H:].sum() / tot)
                rec = float(hist_nb[j, ci_, H:].sum() / totn)
                pts.append({"H": H, "recall": rec, "cand_frac": cand})
            res["scaled"][str(S)][str(al)] = {
                "points": pts, "hits_per_ray_full": float(hits_ray[j, :S].sum() / (nq * S)) * scale}
        res["bound"][str(S)] = {str(m): {"cand_frac": float(b_pass[j, ci_] / nq), "recall": float(b_recall[j, ci_] / nq),
                                         "hits_per_ray_full": float(b_hits[j, :S].sum() / (nq * S)) * scale}
                                for j, m in enumerate(BOUND_M)}
    return res


def best_at(res, S, target):
    best = None
    for al, d in res["scaled"][str(S)].items():
        for p in d["points"]:
            if p["recall"] >= target and (best is None or p["cand_frac"] < best[0]):
                best = (p["cand_frac"], al, p["H"], d["hits_per_ray_full"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--n_queries", type=int, default=300)
    ap.add_argument("--doc_tokens", type=int, default=2_000_000)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = C.dataset(args.dataset)
    eng = Engines(ds, load_ragrt=False)
    R = torch.from_numpy(np.load(os.path.join(ds["sparse_csr_dir"], "svd_rotation_128_to_96.npy"))).cuda().float()
    C96 = torch.from_numpy(np.load(os.path.join(ds["sparse_csr_dir"], "centroids_96d_svd.npy"))).cuda().float()

    qs = E.load_questions(ds["questions_path"])
    tune, _ = E.make_split([q for q, _ in qs], C.TUNE_SIZE, C.SPLIT_SEED)
    text = dict(qs)
    toks = []
    for q in tune[:args.n_queries]:
        Qf, n = eng.encode(text[q])
        toks.append(torch.nn.functional.normalize(Qf[:n].float() @ R, dim=-1))
    half = len(toks) // 2
    Qcal, Qev = torch.cat(toks[:half]), torch.cat(toks[half:])
    print(f"{len(toks)} queries: {len(Qcal):,} calibration / {len(Qev):,} evaluation query tokens")

    total = int(eng.doc_offsets[-1])
    rng = np.random.default_rng(args.seed)
    idx = torch.from_numpy(np.sort(rng.choice(total, size=min(args.doc_tokens, total), replace=False)))
    emb = eng.ranker.embeddings
    D = []
    for a in range(0, len(idx), 500_000):
        b = idx[a:a + 500_000]
        D.append(eng.ranker.codec.decompress(ResidualEmbeddings(emb.codes[b], emb.residuals[b])).cuda().float() @ R)
    D = torch.nn.functional.normalize(torch.cat(D), dim=-1)
    print(f"sampled {len(D):,} of {total:,} doc tokens; {len(C96):,} centroids")

    out = {"doc_tokens": {}, "centroids": {}}
    for K in (10, 100):
        out["doc_tokens"][str(K)] = run_target("doc_tokens", Qcal, Qev, D, K, total, args.chunk)
    for K in (8, 32):
        out["centroids"][str(K)] = run_target("centroids", Qcal, Qev, C96, K, len(C96), args.chunk * 4)

    print("\nSmallest candidate fraction reaching each recall of the true top-K (scaled radii, best alpha and H),")
    print("and the lossless bound with an oracle radius. 'hits/ray' = points inside one slice's sphere,")
    print("i.e. the RT work per ray, at full corpus scale.")
    ref_tok = 20_000 / total
    for tgt, ks in out.items():
        for K, res in ks.items():
            print(f"\n[{tgt}] K={K}  (N={res['full_size']:,}" +
                  (f"; RAGRT today touches ~20K postings per query token = {ref_tok:.1e} of tokens)" if tgt == "doc_tokens" else ")"))
            for S in S_LIST:
                row = f"  S={S:<2} ({3 * S:>2}D) "
                for t in (0.90, 0.95):
                    b = best_at(res, S, t)
                    row += (f"| R>={t:.2f}: cand {b[0]:.2e} (a={b[1]}, H={b[2]}) hits/ray {b[3]:,.0f} " if b
                            else f"| R>={t:.2f}: unreachable ")
                bd = res["bound"][str(S)]
                bm = min(bd.items(), key=lambda kv: kv[1]["cand_frac"])
                row += f"| bound m={bm[0]}: cand {bm[1]['cand_frac']:.2e} recall {bm[1]['recall']:.3f} hits/ray {bm[1]['hits_per_ray_full']:,.0f}"
                print(row, flush=True)
    os.makedirs(ds["results_dir"], exist_ok=True)
    path = os.path.join(ds["results_dir"], "diag_hitcount.json")
    E.dump_json(out, path)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
