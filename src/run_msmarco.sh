#!/bin/bash
# Second dataset end to end: MS MARCO dev small (8.8M passages). Each step is skipped when its
# output exists. Run in tmux; the ground truth and the sweep take hours.
#   ./run_msmarco.sh [check|index|gt|bench|attribution] ...   (default: all, in order)
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
STEPS="${*:-check index gt bench attribution}"
ROOT="${RAGRT_RESULTS:-/local/scratch/a/cashman3/ragrt_results}/msmarco"
CSR=$(python3 -c "import eval_config as C; print(C.dataset('msmarco')['sparse_csr_dir'])")
INDEX=$(python3 -c "import eval_config as C; print(C.dataset('msmarco')['index'])")
SUB=v1

for st in $STEPS; do
  echo "================ $st"
  case "$st" in
    check)
      [ -d "$INDEX" ] || { echo "ColBERT index $INDEX missing: python3 build_msmarco_colbert_index.py (GPU, hours)"; exit 1; }
      du -sh "$INDEX" 2>/dev/null || true; df -h /local/scratch/a | tail -1
      # GPU memory at search time: ColBERT codes (int32) + residuals + the RAGRT CSR.
      python3 - "$INDEX" "$CSR" <<'PY'
import os, sys, glob
idx, csr = sys.argv[1:]
res = sum(os.path.getsize(f) for f in glob.glob(os.path.join(idx, "*.residuals.pt")))
tokens = res / 32                           # 2-bit residuals: 128 dims x 2 bits = 32 bytes per token
c = sum(os.path.getsize(os.path.join(csr, f)) for f in os.listdir(csr) if f.startswith("csr_")) if os.path.isdir(csr) else 0
tot = (res + 4 * tokens + c) / 2**30       # residuals + int32 codes + RAGRT CSR
print(f"~{tokens/1e6:,.0f}M tokens: residuals {res/2**30:.1f} GiB, codes {4*tokens/2**30:.1f} GiB, "
      f"RAGRT CSR {c/2**30:.1f} GiB -> ~{tot:.1f} GiB on the GPU before working buffers (48 GiB card)")
if tot > 42:
    print("WARNING: likely too large for one 48 GB GPU; consider top_m 6-8 for MS MARCO")
PY
      ;;
    index)
      if [ -f "$CSR/csr_packed_24.npy" ] && python3 check_index.py "$CSR"; then
        echo "RAGRT index ok: $CSR"
      else
        ./build_msmarco_sparse_csr.sh
        python3 check_index.py "$CSR"
      fi ;;
    gt)
      python3 build_ground_truth.py --dataset msmarco ;;     # resumes; a no-op when complete
    bench)
      python3 run_benchmarks.py --dataset msmarco --results_subdir "$SUB" \
          --engines plaid,ptms,ragrt,ragrt_bf,ragrt_nofilter
      python3 plot_results.py --dataset msmarco --results_subdir "$SUB" ;;
    attribution)
      ./run_attribution.sh msmarco "$SUB" centroid profile ivf ;;
    *) echo "unknown step $st"; exit 1 ;;
  esac
done
