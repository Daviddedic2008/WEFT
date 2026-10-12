#!/bin/bash
# Route-1 attribution experiments for one dataset, on top of an existing main run (BASE).
#   ./run_attribution.sh lotte v3_ablation [step ...]
# Steps (default: all, in this order; each is resumable and merges into results.json):
#   centroid    ragrt_centroid (no codeword score term) into BASE
#   profile     breakdown + throughput + torch.profiler attribution for every engine in BASE
#   ivf         build the plain-IVF index (once) and run ragrt_ivf against it -> BASE_ivf
#   weft        WEFT (plain IVF + native fp16 prep + imputation) and its no-imputation ablation -> BASE_weft
#   muvera      build MUVERA FDEs (once; needs export_embeddings.py output) and run muvera -> BASE_muvera
# Disk: plain IVF ~ 3 B x (deduplicated postings), MUVERA 2560 dims ~ 12.4 GB on LoTTE.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
DATASET="$1"; BASE="$2"; shift 2
STEPS="${*:-centroid profile ivf weft muvera}"
ROOT="${RAGRT_RESULTS:-/local/scratch/a/cashman3/ragrt_results}/$DATASET"
SCR=/local/scratch/a/cashman3
cfg() { python3 -c "import eval_config as C; print(C.dataset('$DATASET')['$1'])"; }
INDEX=$(cfg index); COLL=$(cfg collection); CSR=$(cfg sparse_csr_dir)
IVF_DIR="$SCR/${DATASET}_plain_ivf"
EMB_DIR="$SCR/${DATASET}_doc_embs"
MUV_DIR="$SCR/muvera_${DATASET}_r20_k4_p8"

for st in $STEPS; do
  echo "================ $st"
  case "$st" in
    centroid)
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "$BASE" --engines ragrt_centroid \
          --stages sweep,select,test,breakdown,drops ;;
    profile)
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "$BASE" \
          --engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid --stages breakdown,throughput,attribution ;;
    ivf)
      [ -f "$IVF_DIR/csr_packed_24.npy" ] || python3 build_plain_ivf.py --index "$INDEX" --collection "$COLL" \
          --src_dir "$CSR" --outdir "$IVF_DIR"
      python3 check_index.py "$IVF_DIR"
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_ivf" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$ROOT/$BASE" --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid \
          --engines ragrt_ivf --stages sweep,select,test,breakdown,throughput,attribution,drops
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_ivf" ;;
    weft)
      [ -f "$IVF_DIR/csr_packed_24.npy" ] || { echo "run the ivf step first"; exit 1; }
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_weft" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$ROOT/${BASE}_ivf" --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf \
          --engines weft,weft_noimpute --stages sweep,select,test,breakdown,throughput,attribution,filtered,drops
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_weft" ;;
    muvera)
      [ -f "$EMB_DIR/doc_embs_fp16.npy" ] || { echo "run export_embeddings.py --dataset $DATASET --outdir $EMB_DIR first"; exit 1; }
      [ -f "$MUV_DIR/fde_docs_fp16.npy" ] || python3 build_muvera_fde.py --emb_dir "$EMB_DIR" --outdir "$MUV_DIR" \
          --reps 20 --k_sim 4 --d_proj 8
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_muvera" \
          --baseline_from "$ROOT/$BASE" --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid \
          --engines muvera --muvera_dir "$MUV_DIR" --stages sweep,select,test,breakdown,throughput,attribution
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_muvera" ;;
    *) echo "unknown step $st"; exit 1 ;;
  esac
done
python3 plot_results.py --dataset "$DATASET" --results_subdir "$BASE"
echo "Attribution tables: $ROOT/$BASE*/summary.md"
