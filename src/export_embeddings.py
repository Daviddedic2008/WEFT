"""
Export what an external engine (FastPLAID) needs to be compared on equal terms (GPU, main env):

  <outdir>/doc_embs_fp16.npy   float16 [tokens, 128]  ColBERT's decompressed doc token vectors,
                               the exact vectors the exact-MaxSim ground truth was computed on
  <outdir>/doc_offsets.npy     int64 [passages + 1]   token range of each passage
  <outdir>/queries.pt          {"qids": [...], "Q": float16 [n, 32, 128], "ntok": [...]} for every
                               query that has ground truth, encoded by the same ColBERT checkpoint

The embedding file is about 68 GB for LoTTE (266M tokens); the script refuses to start without
that much free space plus 10 GB.

  python3 export_embeddings.py --dataset lotte --outdir /local/scratch/a/cashman3/lotte_doc_embs
"""
import os, sys, shutil, argparse, time
import numpy as np
import torch

import eval_config as C
import eval_lib as E
from engines import Engines
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings

CHUNK = 2_000_000


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASETS))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--skip_docs", action="store_true", help="queries only")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ds = C.dataset(args.dataset)
    eng = Engines(ds, load_ragrt=False)

    offsets = eng.doc_offsets.cpu().numpy().astype(np.int64)
    total = int(offsets[-1])
    np.save(os.path.join(args.outdir, "doc_offsets.npy"), offsets)

    if not args.skip_docs:
        need = total * 128 * 2
        free = shutil.disk_usage(args.outdir).free
        if free < need + 10 * 2 ** 30:
            sys.exit(f"FATAL: need {need / 2**30:.1f} GiB + 10 GiB free in {args.outdir}, have {free / 2**30:.1f} GiB")
        path = os.path.join(args.outdir, "doc_embs_fp16.npy")
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(total, 128))
        emb = eng.ranker.embeddings
        t0 = time.time()
        for a in range(0, total, CHUNK):
            b = min(a + CHUNK, total)
            x = eng.ranker.codec.decompress(ResidualEmbeddings(emb.codes[a:b], emb.residuals[a:b]))
            out[a:b] = x.to(torch.float16).cpu().numpy()
            if (a // CHUNK) % 10 == 0 or b == total:
                print(f"  {b:,} / {total:,} tokens ({time.time() - t0:.0f} s)", flush=True)
        out.flush(); del out
        print(f"Wrote {path}")

    z = np.load(ds["gt_path"])
    gt_qids = [str(q) for q in z["qids"]]
    text = dict(E.load_questions(ds["questions_path"]))
    Q, ntok = [], []
    for q in gt_qids:
        Qf, n = eng.encode(text[q])
        Q.append(Qf.cpu()); ntok.append(n)
    torch.save({"qids": gt_qids, "Q": torch.stack(Q), "ntok": ntok}, os.path.join(args.outdir, "queries.pt"))
    print(f"Wrote {len(gt_qids):,} encoded queries to {args.outdir}/queries.pt")


if __name__ == "__main__":
    main()
