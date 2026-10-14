#!/bin/bash
# JUNO's RT-core LUT stage vs the same selection on CUDA cores (juno_bench.cu), on this GPU.
#   ./run_juno.sh [build] [data] [bench]        (default: all three)
# Run with the GPU otherwise idle (no WARP indexing or benchmarks at the same time).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCR=/local/scratch/a/cashman3
OPTIX_PATH="${OPTIX_PATH:-/home/min/a/cashman3/optix_sdk}"
OUT="${RAGRT_RESULTS:-$SCR/ragrt_results}/juno"
DATA=$SCR/juno_data
SIFT=$SCR/sift
ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
STEPS="${*:-build data bench}"
mkdir -p "$OUT"
for st in $STEPS; do
  echo "================ juno $st"
  case "$st" in
    build)
      nvcc -ptx -O3 -arch="$ARCH" -I"$OPTIX_PATH/include" -o "$HERE/juno_rt_device.ptx" "$HERE/juno_rt_device.cu"
      nvcc -O3 -std=c++17 -arch="$ARCH" -I"$OPTIX_PATH/include" -o "$HERE/juno_bench" "$HERE/juno_bench.cu" -lcuda
      echo "built for $ARCH" ;;
    data)
      if [ ! -f "$SIFT/sift_base.fvecs" ]; then
        mkdir -p "$SIFT"
        ( cd "$SIFT" && { wget -q ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz || curl -sSfO ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz; } \
          && tar xzf sift.tar.gz --strip-components=1 && rm -f sift.tar.gz ) || echo "SIFT1M download failed"
      fi
      if [ -f "$SIFT/sift_base.fvecs" ]; then
        python3 "$HERE/prepare_juno_data.py" --data sift1m --sift_dir "$SIFT" --out "$DATA/sift1m"
      fi
      python3 "$HERE/prepare_juno_data.py" --data gaussian --out "$DATA/gaussian" ;;
    bench)
      for d in sift1m gaussian; do
        [ -f "$DATA/$d/meta.txt" ] || continue
        echo "---- $d"
        "$HERE/juno_bench" "$DATA/$d" "$HERE/juno_rt_device.ptx" --nprobe 8,16,32,64 --reps 5 \
          --out "$OUT/juno_rt_vs_cuda_$d.jsonl"
      done ;;
    *) echo "unknown step $st"; exit 1 ;;
  esac
done
