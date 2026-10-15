#!/bin/bash
# Route-1 attribution experiments for one dataset, on top of an existing main run (BASE).
#   ./run_attribution.sh lotte v3_ablation [step ...]
# Steps (default: all, in this order; each is resumable and merges into results.json):
#   centroid    ragrt_centroid (no codeword score term) into BASE
#   profile     breakdown + throughput + torch.profiler attribution for every engine in BASE
#   ivf         build the plain-IVF index (once) and run ragrt_ivf against it -> BASE_ivf
#   weft        RAGRT (plain IVF + native fp16 prep + imputation) and its no-imputation ablation -> BASE_weft
#   resid       residual-correction ablation: build the token-level plain IVF (once; ~1.4x the
#               postings + 4 B/posting token ids) and run weft_tok, weft_resid, weft_resid_norr -> BASE_resid
#   corr        query-correlated topic filters (build_corr_filters.py; rewrites predicates.npy of the
#               index dirs, keeping .bak) + their exact ground truth (build_ground_truth.py --merge)
#   graph       weft (eager, re-run) vs weft_graph (CUDA graphs, one sync) on the plain IVF, with the
#               filtered stage on every filter incl. corr_* for plaid and ptms too -> BASE_graph
#   tue         weft (re-run) vs weft_hash (hash-table Stage 3) and weft_adapt (adaptive probing under
#               filters), filtered stage, and PLAID / PLAID+TMS post-filter budget sweep -> BASE_tue
#   wed         weft (re-run) vs weft_pairs (32-bit pair-sort Stage 3) and the revised weft_adapt
#               (8x cap, 75th-percentile token, cached per-filter counts, fast path) -> BASE_wed
#   plaidfn     PLAID / PLAID+TMS with the filter on Stage 1 (ColBERT filter_fn) vs post-filter and RAGRT,
#               filtered + budget stages on top of the newest figs_* run -> <figs run>_plaidfn
#   encode      stage 0a: stock ColBERT query encoder vs fast_encoder.py (bench_encode.py)
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
TOK_DIR="$SCR/${DATASET}_plain_ivf_tok"
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
    resid)
      [ -f "$TOK_DIR/csr_token_ids.npy" ] || python3 build_plain_ivf.py --index "$INDEX" --collection "$COLL" \
          --src_dir "$CSR" --outdir "$TOK_DIR" --tokens
      python3 check_index.py "$TOK_DIR"
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_resid" --sparse_csr_dir "$TOK_DIR" \
          --baseline_from "$ROOT/${BASE}_weft" \
          --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf,weft,weft_noimpute \
          --core plaid,ptms,weft \
          --engines weft_tok,weft_resid,weft_resid_norr --stages sweep,select,test,breakdown,throughput,attribution,filtered,drops
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_resid" ;;
    corr)
      DIRS=""; for d in "$CSR" "$IVF_DIR" "$TOK_DIR"; do [ -f "$d/predicates.npy" ] && DIRS="$DIRS $d"; done
      python3 build_corr_filters.py --dataset "$DATASET" --update_dirs $DIRS
      python3 build_ground_truth.py --dataset "$DATASET" --merge ;;
    graph)
      [ -f "$IVF_DIR/csr_packed_24.npy" ] || { echo "run the ivf step first"; exit 1; }
      PREV="$ROOT/$BASE"   # newest earlier run that has weft's sweep: BASE_resid, BASE_weft, or BASE itself
      for c in "$ROOT/${BASE}_weft" "$ROOT/${BASE}_resid"; do [ -f "$c/results.json" ] && PREV="$c"; done
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_graph" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$PREV" \
          --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf,weft_noimpute,weft_tok,weft_resid,weft_resid_norr \
          --core plaid,ptms,weft --filtered_extra plaid,ptms \
          --engines weft,weft_graph --stages sweep,select,test,breakdown,throughput,attribution,filtered
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_graph" ;;
    tue)
      [ -f "$IVF_DIR/csr_packed_24.npy" ] || { echo "run the ivf step first"; exit 1; }
      PREV="$ROOT/$BASE"
      for c in "$ROOT/${BASE}_weft" "$ROOT/${BASE}_resid" "$ROOT/${BASE}_graph"; do [ -f "$c/results.json" ] && PREV="$c"; done
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_tue" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$PREV" \
          --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf,weft_noimpute,weft_tok,weft_resid,weft_resid_norr,weft_graph \
          --core plaid,ptms,weft --engines weft,weft_hash,weft_adapt \
          --stages sweep,select,test,breakdown,throughput,attribution,filtered,filter_budget
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_tue" ;;
    wed)
      [ -f "$IVF_DIR/csr_packed_24.npy" ] || { echo "run the ivf step first"; exit 1; }
      PREV="$ROOT/$BASE"
      for c in "$ROOT/${BASE}_weft" "$ROOT/${BASE}_resid" "$ROOT/${BASE}_graph" "$ROOT/${BASE}_tue"; do [ -f "$c/results.json" ] && PREV="$c"; done
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_wed" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$PREV" \
          --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf,weft_noimpute,weft_tok,weft_resid,weft_resid_norr,weft_graph,weft_hash \
          --core plaid,ptms,weft --engines weft,weft_pairs,weft_adapt \
          --stages sweep,select,test,breakdown,throughput,attribution,filtered
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_wed" ;;
    thu)
      [ -f "$IVF_DIR/csr_packed_24.npy" ] || { echo "run the ivf step first"; exit 1; }
      PREV="$ROOT/$BASE"
      for c in "$ROOT/${BASE}_weft" "$ROOT/${BASE}_resid" "$ROOT/${BASE}_graph" "$ROOT/${BASE}_tue" "$ROOT/${BASE}_wed"; do [ -f "$c/results.json" ] && PREV="$c"; done
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_thu" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$PREV" \
          --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf,weft_noimpute,weft_tok,weft_resid,weft_resid_norr,weft_graph,weft_hash,weft_pairs \
          --core plaid,ptms,weft --engines weft,weft_adapt \
          --stages sweep,select,test,attribution,filtered
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_thu" ;;
    oracle)
      [ -f "$IVF_DIR/csr_packed_24.npy" ] || { echo "run the ivf step first"; exit 1; }
      PREV="$ROOT/${BASE}_thu"; [ -f "$PREV/results.json" ] || { echo "run the thu step first"; exit 1; }
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_oracle" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$PREV" \
          --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf,weft_noimpute,weft,weft_tok,weft_resid,weft_resid_norr,weft_graph,weft_hash,weft_pairs,weft_adapt \
          --core plaid,ptms,weft --engines weft_oracle,weft_oracle_adapt --filtered_extra weft,weft_adapt \
          --stages select,filtered
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_oracle" ;;
    oracle2)
      [ -f "$IVF_DIR/csr_packed_24.npy" ] || { echo "run the ivf step first"; exit 1; }
      PREV="$ROOT/${BASE}_thu"; [ -f "$PREV/results.json" ] || { echo "run the thu step first"; exit 1; }
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "${BASE}_oracle2" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$PREV" \
          --import_engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter,ragrt_centroid,ragrt_ivf,weft_noimpute,weft,weft_tok,weft_resid,weft_resid_norr,weft_graph,weft_hash,weft_pairs,weft_adapt \
          --core plaid,ptms,weft --engines weft_oracle,weft_oracle_adapt,weft_oimp,weft_oskip --filtered_extra weft,weft_adapt \
          --stages select,filtered
      python3 plot_results.py --dataset "$DATASET" --results_subdir "${BASE}_oracle2" ;;
    plaidfn)   # PLAID / PLAID+TMS with the filter on the Stage 1 candidates (ColBERT filter_fn) vs post-filter and
               # RAGRT, all re-measured in one run on top of the newest figs_* run (its sweeps and selection)
      PREV=$(ls -dt "$ROOT"/figs_* 2>/dev/null | grep -v _plaidfn | head -1)
      [ -n "$PREV" ] && [ -f "$PREV/results.json" ] || { echo "run run_figures.sh $DATASET first"; exit 1; }
      OUTN="$(basename "$PREV")_plaidfn"
      python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "$OUTN" --sparse_csr_dir "$IVF_DIR" \
          --baseline_from "$PREV" --import_engines plaid,ptms,weft,weft_adapt,muvera \
          --core plaid,ptms,weft --engines plaid_fn,ptms_fn,weft_adapt,weft_oracle_adapt \
          --filtered_extra plaid,ptms,weft --stages select,filtered,filter_budget
      python3 plot_results.py --dataset "$DATASET" --results_subdir "$OUTN"
      python3 make_figures.py --dataset "$DATASET" --main "$OUTN" ;;
    encode)
      python3 bench_encode.py --dataset "$DATASET" --sparse_csr_dir "$IVF_DIR" ;;
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
