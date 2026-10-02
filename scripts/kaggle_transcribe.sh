#!/usr/bin/env bash
# Transcribe every audio/video file in the attached Kaggle dataset, unattended.
# Output: /kaggle/working/results/<name>.txt / .srt / .json  (download from the notebook's Output tab).
#
#   !rm -rf transcribe && git clone -q --depth 1 -b matt/relaxed-meitner-2lfrx6 https://github.com/mattobryan/transcribe.git
#   !cd transcribe && bash scripts/kaggle_transcribe.sh
#
# Optional settings before "bash": MODEL=small|large-v3-turbo (default: large-v3-turbo with a GPU, small without)
#   LANGUAGE=sw|en|auto  SPEAKERS=0
set -euo pipefail
cd "$(dirname "$0")/.."
INPUT=${INPUT:-/kaggle/input}
OUT=${OUT:-/kaggle/working/results}
MODEL=${MODEL:-auto}
LANGUAGE=${LANGUAGE:-sw}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "== [1/3] Installing"
pip install -q "faster-whisper>=1.1,<1.3" "jiwer>=3,<5" "soundfile>=0.12" "numpy<2.3"
pip install -q --no-deps -e .

# CTranslate2 needs cuBLAS/cuDNN; they ship as pip wheels with torch.
NV_LIBS=$(python - <<'PY'
import importlib.util, pathlib
dirs = []
for name in ("nvidia.cublas", "nvidia.cudnn"):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        continue
    if spec and spec.submodule_search_locations:
        lib = pathlib.Path(list(spec.submodule_search_locations)[0]) / "lib"
        if lib.is_dir():
            dirs.append(str(lib))
print(":".join(dirs))
PY
)
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; then
    HAS_GPU=1
else
    HAS_GPU=0
    echo "No GPU (Accelerator is off or unavailable). Continuing on the CPU, which is slower."
fi
# Unless MODEL was set explicitly: the accurate model on a GPU, the small one on a CPU
# (large-v3-turbo on a CPU would take many hours).
if [ "$MODEL" = "auto" ]; then
    if [ "$HAS_GPU" = "1" ]; then MODEL=large-v3-turbo; else MODEL=small; fi
fi

echo "== [2/3] Files found in $INPUT"
find "$INPUT" -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.m4a' -o -iname '*.mp4' -o -iname '*.mkv' \
    -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.avi' -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.flac' \) \
    -printf '  %p  (%s bytes)\n'

echo "== [3/3] Transcribing with $MODEL (language: $LANGUAGE)"
SPEAKER_FLAG=""; [ "${SPEAKERS:-0}" = "1" ] || SPEAKER_FLAG="--no-speakers"
python -m transcribe.app.batch "$INPUT" --out "$OUT" --model "$MODEL" --language "$LANGUAGE" $SPEAKER_FLAG
echo "Done. Download the .txt and .srt files from the Output tab (or results.zip)."
cd "$OUT/.." && zip -q -r results.zip "$(basename "$OUT")" -x "*/sessions/*" || true
