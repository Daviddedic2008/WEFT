#!/bin/bash
# Unattended overnight queue. Each step logs to ~/overnight/<n>_<name>.log; one failure does not stop
# the rest (except: the RAGRT ablations need the rebuilt extension and passing GPU tests).
# Progress: cat ~/overnight/STATUS        Start it in a FRESH tmux window (no venv active):
#   tmux new -s night     then     ~/RTRAG/src/overnight.sh
set -u
cd "$(dirname "$0")"
SRC=$(pwd)
OUT=~/overnight; mkdir -p "$OUT"
SCR=/local/scratch/a/cashman3
FP_PY=$SCR/venvs/fastplaid/bin/python3
export TMPDIR=$SCR/tmp PYTHONUNBUFFERED=1; mkdir -p "$TMPDIR"
STATUS=$OUT/STATUS
N=0
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
echo "== overnight $(date)  git $(git rev-parse --short HEAD)" >> "$STATUS"

# 1. FastPLAID nbits=2: finish the stages the dropped SSH session killed (~20 min)
step fastplaid2 "$FP_PY bench_fastplaid.py --dataset lotte --emb_dir $SCR/lotte_doc_embs \
  --index_dir $SCR/fastplaid_lotte_nbits2 --nbits 2 --compare v3_ablation --stages test,filtered,throughput"

# 2. Rebuild the extension (Stage 2 kernel changed) and run the tests
OK=1
step build "python3 setup_corr_3d.py build_ext --inplace && python3 -c 'import rtrag_corr_3d as X; X.set_codeword_weight(1.0); print(\"extension ok\")'" || OK=0
[ $OK = 1 ] && { step gpu_tests "cd .. && python3 tests/test_gpu_kernels.py" || OK=0; }
step cpu_tests "cd .. && python3 tests/test_ablation_cpu.py && python3 tests/test_harness_smoke.py | tail -3"

# 3. RAGRT attribution: no-codeword-score ablation, profiler, plain IVF on RAGRT kernels (~1-2 h)
if [ $OK = 1 ]; then
  step attribution "./run_attribution.sh lotte v3_ablation centroid profile ivf"
else
  echo "SKIP attribution (build or GPU tests failed)" | tee -a "$STATUS"
fi

# 4. PHRT kill test (independent of RAGRT)
step phrt_setup "cd ~ && { [ -d PHRT ] || git clone git@github.com:cashmanjack/PHRT.git; } && cd PHRT && git pull -q && \
  { python3 -m pip install -q ripser || python3 -m pip install -q --user ripser; } && python3 -m pytest -q tests"
step phrt_full "cd ~/PHRT && python3 scripts/killtest.py --tag full --n 20000 --degrees 8,16,32,64"
step phrt_h2   "cd ~/PHRT && python3 scripts/killtest.py --tag h2 --n 5000 --degrees 8,16 --maxdim 2"
step phrt_bench "cd ~/PHRT && python3 scripts/fetch_benchmarks.py && for f in data/*/*; do \
  python3 scripts/killtest.py --tag bench_\$(basename \$f .txt) --datasets file:\$f --n 5000 --degrees 8,32 || true; done"

# 5. FastPLAID nbits=4: is its 0.933 recall ceiling double quantization? (~45 min; frees the nbits=2 index first)
step fastplaid4 "rm -rf $SCR/fastplaid_lotte_nbits2 && $FP_PY bench_fastplaid.py --dataset lotte \
  --emb_dir $SCR/lotte_doc_embs --index_dir $SCR/fastplaid_lotte_nbits4 --nbits 4 --compare v3_ablation"

# 6. MUVERA (needs ~13 GB: the nbits=4 index is removed first; its results are already saved)
if [ $OK = 1 ]; then
  step muvera "rm -rf $SCR/fastplaid_lotte_nbits4 && ./run_attribution.sh lotte v3_ablation muvera"
fi

# 7. MS MARCO: only the memory/disk check (decide in the morning)
step msmarco_check "./run_msmarco.sh check"

echo "== finished $(date)" | tee -a "$STATUS"
