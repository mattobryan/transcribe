#!/usr/bin/env bash
# One-shot pilot on Kaggle: install, run the tests, run every recording found in
# /kaggle/input, and zip the reports into /kaggle/working/pilot_reports.zip.
#
#   !git clone -q --depth 1 -b matt/relaxed-meitner-2lfrx6 https://github.com/mattobryan/transcribe.git
#   !cd transcribe && bash scripts/kaggle_pilot.sh
#
# Settings (environment variables, all optional):
#   INPUT=/kaggle/input  OUT=/kaggle/working/pilot
#   MODELS=small,large-v3-turbo  LANGS=sw,en,auto  MAX_SEGMENTS=0 (all)  ONLY=<recording ids>
set -euo pipefail
cd "$(dirname "$0")/.."

INPUT=${INPUT:-/kaggle/input}
OUT=${OUT:-/kaggle/working/pilot}
MODELS=${MODELS:-small,large-v3-turbo}
LANGS=${LANGS:-sw,en,auto}
MAX_SEGMENTS=${MAX_SEGMENTS:-0}
ONLY=${ONLY:-}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/kaggle_run.log") 2>&1

echo "== [1/5] Installing dependencies"
pip install -q "transformers>=4.40,<5" "faster-whisper>=1.1,<1.3" "jiwer>=3.0,<5" \
    "python-docx>=1.1,<2" "pypdf>=5,<7" "librosa>=0.10,<0.12" "soundfile>=0.12,<0.15" pytest

# CTranslate2 (faster-whisper) needs cuBLAS/cuDNN; Kaggle ships them as pip wheels.
NV_LIBS=$(python - <<'PY'
import importlib, os
paths = []
for name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        module = importlib.import_module(name)
        paths.append(os.path.dirname(module.__file__) if getattr(module, "__file__", None) else list(module.__path__)[0])
    except Exception:
        pass
print(":".join(paths))
PY
)
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "== [2/5] Environment"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU (enable one under Settings > Accelerator)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== [3/5] Tests"
python -m pytest tests/pilot -q

echo "== [4/5] Whisper GPU preflight"
WHISPER_DEVICE=cpu
if python - <<'PY'
import numpy as np
from faster_whisper import WhisperModel
model = WhisperModel("tiny", device="cuda", compute_type="float16")
list(model.transcribe(np.zeros(16000, dtype=np.float32), language="sw")[0])
PY
then WHISPER_DEVICE=cuda; fi
echo "Whisper device: $WHISPER_DEVICE"

echo "== [5/5] Pilot on every recording in $INPUT"
STATUS=0
python -m src.transcribe.pilot.batch --input "$INPUT" --out "$OUT" --only "$ONLY" \
    --models "$MODELS" --languages "$LANGS" --max-segments "$MAX_SEGMENTS" \
    --whisper-device "$WHISPER_DEVICE" || STATUS=$?

cd "$OUT/.."
zip -q -r pilot_reports.zip "$(basename "$OUT")" \
    -i "*summary.md" "*summary.json" "*report.md" "*report.json" \
       "*alignment.json" "*segments.json" "*kaggle_run.log" || true
echo "Done (exit $STATUS). Download /kaggle/working/pilot_reports.zip and send summary.md + report.md files."
exit $STATUS
