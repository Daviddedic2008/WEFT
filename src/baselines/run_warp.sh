#!/bin/bash
# WARP + XTR baseline on LoTTE pooled dev (CPU retrieval, GPU indexing), isolated from the RAGRT env.
#   ./run_warp.sh [setup] [index] [search]        (default: all three)
# setup : clone xtr-warp into $SCR/xtr-warp, venv $SCR/venvs/warp (inherits the system torch),
#         pip deps (warp imports onnxruntime, line_profiler, pytrec_eval at import time; its OpenVINO
#         import is patched to be optional),
#         pip deps, .env (INDEX_ROOT, EXPERIMENT_ROOT, LOTTE_COLLECTION_PATH)
# index : google/xtr-base-en index of lotte/pooled/dev, nbits=2 (~1 to 3 h; needs ~30 GB scratch)
# search: nprobe 8,16,32 x threads 1,16 -> $RAGRT_RESULTS/lotte/warp_lotte_pooled.json
set -euo pipefail
export PYTHONUNBUFFERED=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
HERE="$(cd "$(dirname "$0")" && pwd)"
SCR=/local/scratch/a/cashman3
REPO=$SCR/xtr-warp; VENV=$SCR/venvs/warp; LOTTE=$SCR/lotte
OUT="${RAGRT_RESULTS:-$SCR/ragrt_results}/lotte"
export TMPDIR=$SCR/tmp; mkdir -p "$TMPDIR"
STEPS="${*:-setup index search}"
for st in $STEPS; do
  echo "================ warp $st"
  case "$st" in
    setup)
      [ -d "$LOTTE/pooled/dev" ] || { echo "FATAL: $LOTTE/pooled/dev missing (official LoTTE tarball)"; exit 1; }
      [ -d "$REPO" ] || git clone -q https://github.com/jlscheerer/xtr-warp.git "$REPO"
      [ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
      "$VENV/bin/pip" install -q --no-cache-dir bitarray ujson python-dotenv jsonlines gitpython psutil ninja \
          faiss-cpu beir scipy tqdm transformers sentencepiece huggingface_hub \
          onnxruntime line_profiler pytrec-eval-terrier pandas matplotlib datasets tabulate
      # WARP imports its OpenVINO runtime at import time; we use the default torch runtime, so make it optional
      "$VENV/bin/python" - "$REPO/warp/engine/runtime/openvino_model.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "import openvino.runtime as ov\nfrom optimum.intel.openvino import OVModelForSeq2SeqLM\n"
if a in s:
    s = s.replace(a, "try:  # patched by RAGRT run_warp.sh: the OpenVINO runtime is not used\n"
                     "    import openvino.runtime as ov\n    from optimum.intel.openvino import OVModelForSeq2SeqLM\n"
                     "except ImportError:\n    ov = OVModelForSeq2SeqLM = None\n")
    open(p, "w").write(s); print("patched", p)
PY
      mkdir -p $SCR/warp_indexes $SCR/warp_experiments
      printf "INDEX_ROOT=%s\nEXPERIMENT_ROOT=%s\nBEIR_COLLECTION_PATH=%s\nLOTTE_COLLECTION_PATH=%s\n" \
          $SCR/warp_indexes $SCR/warp_experiments $SCR/beir "$LOTTE" > "$REPO/.env"
      (cd "$REPO" && PYTHONPATH="$REPO" "$VENV/bin/python" -c "import torch, faiss, warp, warp.engine.config, warp.engine.searcher, warp.data.queries, warp.engine.utils.collection_indexer, warp.engine.utils.index_converter; print('warp import ok, torch', torch.__version__)") ;;
    index)
      free=$(df -BG --output=avail $SCR | tail -1 | tr -dc 0-9)
      [ "$free" -ge 30 ] || { echo "FATAL: only ${free} GB free on scratch (need ~30)"; exit 1; }
      cd "$REPO" && PYTHONPATH="$REPO" "$VENV/bin/python" "$HERE/warp_lotte.py" index --nbits 2 --out "$OUT" ;;
    search)
      cd "$REPO" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" "$VENV/bin/python" "$HERE/warp_lotte.py" search \
          --nbits 2 --nprobe 8,16,32 --threads 1,16 --out "$OUT" ;;
    *) echo "unknown step $st"; exit 1 ;;
  esac
done
