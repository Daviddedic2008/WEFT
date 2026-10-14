#!/bin/bash
# Fresh measurements, then figures: every engine in the figures is measured in one new run, on the
# current code and GPU, with no results imported from earlier runs (no --baseline_from).
#   ./run_figures.sh lotte            (GPU must be otherwise idle; ~hours on LoTTE)
#   ./run_figures.sh msmarco
# Steps:
#   1. run_benchmarks.py -> <results>/<ds>/figs_<date>: PLAID, PLAID+TMS, RAGRT (weft) and its filter
#      variants (adaptive; filter-aware and its two ablation halves in the filtered stage), plus MUVERA if its FDEs exist.
#      Stages: sweep, select, test, breakdown, throughput, attribution, filtered, filter_budget.
#   2. FastPLAID (own venv) re-measured with --compare figs_<date>, if its index and the exported
#      embeddings exist; otherwise skipped and make_figures marks the old FastPLAID results as stale.
#   3. make_figures.py on the new run. WARP (CPU, own venv) is not re-run here: run_warp.sh search.
# The RT-core design needs the codebook index, so it cannot share this run and is left out.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
DATASET="$1"
TAG="${2:-figs_$(date +%m%d_%H%M)}"
SCR=/local/scratch/a/cashman3
IVF_DIR="$SCR/${DATASET}_plain_ivf"
EMB_DIR="$SCR/${DATASET}_doc_embs"
MUV_DIR="$SCR/muvera_${DATASET}_r20_k4_p8"
FP_PY=$SCR/venvs/fastplaid/bin/python3
FP_IDX="$SCR/fastplaid_${DATASET}_nbits2"

[ -f "$IVF_DIR/csr_packed_24.npy" ] || { echo "FATAL: no plain IVF index at $IVF_DIR (run_attribution.sh $DATASET <base> ivf)"; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "FATAL: another process is using the GPU; timings would be wrong"; nvidia-smi; exit 1
fi

ENGINES=plaid,ptms,weft,weft_adapt
# filter-aware variants reuse weft / weft_adapt's selected params (PARAMS_FROM), so they run in the filtered stage only
ORACLE=weft_oracle,weft_oracle_adapt,weft_oimp,weft_oskip
MUV_ARGS=""
if [ -f "$MUV_DIR/fde_docs_fp16.npy" ]; then ENGINES="$ENGINES,muvera"; MUV_ARGS="--muvera_dir $MUV_DIR"
else echo "note: no MUVERA FDEs at $MUV_DIR; MUVERA left out of this run"; fi

echo "================ main run -> $TAG ($(git rev-parse --short HEAD))"
python3 run_benchmarks.py --dataset "$DATASET" --results_subdir "$TAG" --sparse_csr_dir "$IVF_DIR" \
    --core plaid,ptms,weft --engines "$ENGINES" --filtered_extra "$ORACLE" $MUV_ARGS \
    --stages sweep,select,test,breakdown,throughput,attribution,filtered,filter_budget

echo "================ FastPLAID"
if [ -x "$FP_PY" ] && [ -f "$EMB_DIR/doc_embs_fp16.npy" ]; then
  STAGES=sweep,select,test,filtered,throughput
  [ -d "$FP_IDX" ] || STAGES="build,$STAGES"
  "$FP_PY" bench_fastplaid.py --dataset "$DATASET" --emb_dir "$EMB_DIR" --index_dir "$FP_IDX" --nbits 2 \
      --compare "$TAG" --stages "$STAGES"
else
  echo "skipped: needs $FP_PY and $EMB_DIR (export_embeddings.py); old FastPLAID results will be marked stale"
fi

echo "================ figures"
python3 make_figures.py --dataset "$DATASET" --main "$TAG"
echo "done: ${RAGRT_RESULTS:-$SCR/ragrt_results}/$DATASET/$TAG/figures"
