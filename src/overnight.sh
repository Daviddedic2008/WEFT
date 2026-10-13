#!/bin/bash
# Unattended run queue. Each step logs to ~/overnight/<n>_<name>.log; a failure does not stop the
# rest (the RAGRT ablations and MUVERA need the extension to import and the GPU tests to pass).
#   tmux new -s night;  ~/RTRAG/src/overnight.sh [step ...]
# Steps (default: build gpu_tests attribution phrt muvera fastplaid4):
#   fastplaid2 build gpu_tests cpu_tests attribution phrt muvera fastplaid4 msmarco_check
#   lotte_corr lotte_graph msmarco_encode msmarco_corr msmarco_graph lotte_tue msmarco_tue
#   lotte_wed msmarco_wed warp_setup warp_index warp_search
# Progress: cat ~/overnight/STATUS
set -u
cd "$(dirname "$0")"
SRC=$(pwd)
OUT=~/overnight; mkdir -p "$OUT"
SCR=/local/scratch/a/cashman3
FP_PY=$SCR/venvs/fastplaid/bin/python3
export TMPDIR=$SCR/tmp PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; mkdir -p "$TMPDIR"
STATUS=$OUT/STATUS
STEPS="${*:-build gpu_tests attribution phrt muvera fastplaid4}"
N=$(ls "$OUT"/*.log 2>/dev/null | wc -l)
step() {   # step <name> <command...>
  N=$((N + 1)); local name="$1"; shift
  local log="$OUT/$(printf %02d $N)_$name.log" t0=$(date +%s)
  echo "$(date '+%H:%M') START $name" | tee -a "$STATUS"
  ( cd "$SRC" && eval "$@" ) > "$log" 2>&1
  local rc=$? dt=$(( ($(date +%s) - t0) / 60 ))
  echo "$(date '+%H:%M') $([ $rc -eq 0 ] && echo DONE || echo "FAIL rc=$rc") $name (${dt} min) -> $log" | tee -a "$STATUS"
  df -h /local/scratch/a | tail -1 >> "$STATUS"
  return $rc
}
free_gb() { df -BG --output=avail /local/scratch/a | tail -1 | tr -dc 0-9; }
echo "== $(date)  git $(git rev-parse --short HEAD)  steps: $STEPS" >> "$STATUS"
OK=1; BOK=1   # OK: build + GPU tests passed; BOK: build passed
for st in $STEPS; do case "$st" in
  fastplaid2)
    step fastplaid2 "$FP_PY bench_fastplaid.py --dataset lotte --emb_dir $SCR/lotte_doc_embs \
      --index_dir $SCR/fastplaid_lotte_nbits2 --nbits 2 --compare v3_ablation --stages test,filtered,throughput" ;;
  build)   # torch must be imported first: it loads libc10 for the extension
    step build "python3 setup_corr_3d.py build_ext --inplace > /dev/null && \
      python3 -c 'import torch, rtrag_corr_3d as X; X.set_codeword_weight(1.0); print(\"extension ok\", X.__file__)'" || { OK=0; BOK=0; } ;;
  gpu_tests)
    [ $OK = 1 ] && { step gpu_tests "cd .. && python3 tests/test_gpu_kernels.py" || OK=0; } ;;
  cpu_tests)
    step cpu_tests "cd .. && python3 tests/test_ablation_cpu.py && python3 tests/test_harness_smoke.py | tail -3" ;;
  attribution)
    if [ $OK = 1 ]; then step attribution "./run_attribution.sh lotte v3_ablation centroid profile ivf"
    else echo "SKIP attribution (build or GPU tests failed)" | tee -a "$STATUS"; fi ;;
  phrt)
    step phrt_setup "cd ~ && { [ -d PHRT ] || git clone git@github.com:cashmanjack/PHRT.git; } && cd PHRT && git pull -q && \
      { python3 -c 'import ripser' 2>/dev/null || python3 -m pip install -q --user ripser; } && python3 -m pytest -q tests" && {
    step phrt_full "cd ~/PHRT && python3 scripts/killtest.py --tag full --n 20000 --degrees 8,16,32,64"
    step phrt_h2   "cd ~/PHRT && python3 scripts/killtest.py --tag h2 --n 5000 --degrees 8,16 --maxdim 2"
    step phrt_bench "cd ~/PHRT && python3 scripts/fetch_benchmarks.py && fails=0 && for f in data/*/*; do \
      python3 scripts/killtest.py --tag bench_\$(basename \$f | cut -d. -f1) --datasets file:\$f --n 5000 --degrees 8,32 \
        || { echo \"!! \$f failed\"; fails=\$((fails+1)); }; done; echo \"\$fails files failed\""; } ;;
  muvera)
    if [ $OK = 1 ]; then step muvera "./run_attribution.sh lotte v3_ablation muvera"
    else echo "SKIP muvera (build or GPU tests failed)" | tee -a "$STATUS"; fi ;;
  fastplaid4)   # the nbits=4 build needs ~25 GB of scratch at its peak
    if [ "$(free_gb)" -ge 35 ]; then
      step fastplaid4 "$FP_PY bench_fastplaid.py --dataset lotte --emb_dir $SCR/lotte_doc_embs \
        --index_dir $SCR/fastplaid_lotte_nbits4 --nbits 4 --compare v3_ablation; rm -rf $SCR/fastplaid_lotte_nbits4"
    else echo "SKIP fastplaid4: only $(free_gb) GB free (needs ~35)" | tee -a "$STATUS"; fi ;;
  msmarco_check) step msmarco_check "./run_msmarco.sh check" ;;
  # Oct 12 queue: correlated filters + their ground truth, CUDA-graph WEFT, fast query encoder
  lotte_corr)     step lotte_corr     "./run_attribution.sh lotte v3_ablation corr" ;;
  lotte_graph)    [ $OK = 1 ] && step lotte_graph "./run_attribution.sh lotte v3_ablation graph" ;;
  msmarco_encode) step msmarco_encode "./run_attribution.sh msmarco v1 encode" ;;
  msmarco_corr)   step msmarco_corr   "./run_attribution.sh msmarco v1 corr" ;;
  msmarco_graph)  [ $OK = 1 ] && step msmarco_graph "./run_attribution.sh msmarco v1 graph" ;;
  # Oct 13 queue: hash Stage 3, adaptive probing, post-filter budget sweep
  lotte_tue)      [ $BOK = 1 ] && step lotte_tue   "./run_attribution.sh lotte v3_ablation tue" ;;
  msmarco_tue)    [ $BOK = 1 ] && step msmarco_tue "./run_attribution.sh msmarco v1 tue" ;;
  # Oct 13 night: pair-sort Stage 3, revised adaptive probing, WARP + XTR baseline (own venv)
  lotte_wed)      [ $BOK = 1 ] && step lotte_wed   "./run_attribution.sh lotte v3_ablation wed" ;;
  msmarco_wed)    [ $BOK = 1 ] && step msmarco_wed "./run_attribution.sh msmarco v1 wed" ;;
  warp_setup)     step warp_setup  "./baselines/run_warp.sh setup" ;;
  warp_index)     step warp_index  "./baselines/run_warp.sh index" ;;
  warp_search)    step warp_search "./baselines/run_warp.sh search" ;;
  *) echo "unknown step $st" | tee -a "$STATUS" ;;
esac; done
echo "== finished $(date)" | tee -a "$STATUS"
