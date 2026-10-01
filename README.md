# English-Swahili Code-Switching Transcriber

A custom speech-to-text transcriber for English-Swahili code-switching speech patterns commonly found in Kenya.

**Planning documents** (problem statement, PRD, TRD, UI/UX, app flow, backend schema,
implementation plan) are in [docs/](docs/README.md). They supersede the original
[docs/PROJECT_WORKFLOW.md](docs/PROJECT_WORKFLOW.md).

## Status

- **Planning documents:** problem statement, PRD, TRD, UI/UX, app flow, backend schema,
  implementation plan, Kaggle runbook and style guide, as versioned LaTeX/PDF with change logs.
- **Pilot (done):** aligns a reviewed DOCX transcript to its audio, cuts training-sized
  chunks and measures stock Whisper on them. See the TRD section on long-form alignment and the Plan's *Pilot* section.
- **Phase 0 (in progress):** package layout, legacy baseline moved aside, grouped splits, CI.
- **Tasks:** tracked in [GitHub Issues](https://github.com/mattobryan/transcribe/issues),
  labelled `owner:you` / `owner:claude` and by phase.

The product direction is a fine-tuned multilingual Whisper (TRD §2). The original from-scratch
MFCC CNN-LSTM models are kept as a learning baseline in `transcribe.legacy`.

## Project Structure

```
transcribe/
|-- docs/                     # LaTeX sources (latex/) and PDFs (pdf/, pdf/archive/)
|-- src/transcribe/
|   |-- pilot/                # alignment, segmentation, zero-shot Whisper, reports, Kaggle batch
|   |-- preprocessing/        # DOCX/PDF transcript parsers, energy VAD, chunk export, manifests
|   |-- app/                  # transcription app: upload, background transcription, chunk editor
|   |-- script/               # CLIs: parse_transcript, prepare_audio, review, export_training_manifest
|   `-- legacy/               # from-scratch CTC/Seq2Seq baseline: train, evaluate, infer
|-- tests/                    # pilot/ and legacy/ test suites
|-- scripts/kaggle_pilot.sh   # one-shot Kaggle run
|-- notebooks/                # Kaggle notebook
|-- requirements/             # dependency groups: core, pilot, ui, legacy, dev
`-- .github/workflows/ci.yml  # lint + tests on every push
```

## Installation

```bash
python -m pip install -r requirements-pilot.txt   # pilot + review UI (installs the package too)
python -m pip install -r requirements.txt         # legacy baseline only
python -m pip install -r requirements/dev.txt     # tests and lint
```

Each requirements file ends with `-e .`, which installs this repository as the `transcribe`
package, so every tool runs as `python -m transcribe....` from any folder. Run them from the
repository root.

> **Use `python -m pip`, not a bare `pip`.** On many Windows systems a bare `pip` points to a
> *different* Python installation than the `python` that runs this project, so packages you
> install with it are not visible. `python -m pip` always targets the interpreter you are using.

Run the checks the same way CI does:

```bash
ruff check .
python -m pytest -q
```

## Transcription app

```bash
python -m pip install -r requirements-pilot.txt     # once
python -m transcribe.app                            # opens http://127.0.0.1:8000
```

1. **Load audio**: drop or choose any audio or video file, pick the language setting and model,
   and press **Transcribe**.
2. **Transcription in progress**: the audio is split into chunks at pauses (up to 30 s) and
   transcribed with Whisper in the background. You can close the page; it keeps going. Speakers are
   labelled when [pyannote](https://huggingface.co/pyannote/speaker-diarization-3.1) is installed and
   `HF_TOKEN` is set (accept the model's terms on Hugging Face first); otherwise chunks have no label.
3. **Correct**: each chunk plays with its text below it. Moving to another chunk saves your edit and
   plays the new chunk.

| Key | Action |
|---|---|
| `Esc` | Play / pause |
| `F1` / `F2` | Back / forward 3 s |
| `F3` / `F4` | Slower / faster (0.5x to 2x) |
| `Ctrl+Enter` | Save and next (plays automatically) |
| `Alt+Up` / `Alt+Down` | Previous / next chunk |
| `Ctrl+S` | Save |

**How your corrections help while you work** (no retraining; that happens in batches on Kaggle):
- a word you correct the same way twice becomes a rule applied to every chunk you have not edited;
- your recent corrected text and the new words you typed (names, hybrid spellings) are given to
  Whisper as context, and the next 3 chunks are re-transcribed in the background before you reach them.
  Chunks improved this way are marked ↻.

Sessions are kept in `data/app/sessions/<id>/`. Export as TXT, SRT or JSON from the top bar. Every
corrected chunk is training data: `python -m transcribe.db.migrate_json data/app/sessions/*/session.json`.

## Legacy baseline (from-scratch models)

```bash
python -m transcribe.legacy.train --config config/default.yaml
python -m transcribe.legacy.evaluate --checkpoint checkpoints/final_model.pt
python -m transcribe.legacy.infer --audio data/audio/sample1.wav --checkpoint checkpoints/final_model.pt
```

`data/metadata.json` is a list of `{"audio_file": ..., "transcript": ...}`. Splits are grouped by
recording (chunk names `<recording>_000123.wav`, or a `recording_id` field), so no recording is in
two splits, and the vocabulary is built from the training transcripts only.

## Pilot: One Recording (alignment + zero-shot Whisper)

Before training anything, measure one reviewed recording: how well its DOCX transcript
aligns to the audio, and how well stock Whisper already does on it. Everything runs on CPU
and every stage is cached in `--out`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-pilot.txt

# Optional 5-minute smoke test on public FLEURS clips with known answers
# (downloads ~310 MB once; the only suspect turn should be p0008):
.\.venv\Scripts\python.exe -m transcribe.pilot.make_mock --out data/pilot/mock
.\.venv\Scripts\python.exe -m transcribe.pilot.run `
  --audio data/pilot/mock/mock_interview.wav --transcript data/pilot/mock/mock_interview.docx `
  --out data/pilot/mock/run --languages sw

# Quick check on the real recording (about 10 minutes after the one-off alignment step):
.\.venv\Scripts\python.exe -m transcribe.pilot.run `
  --audio "data/audio/NRCCW_ KSM09.MP3" `
  --transcript data/audio/NRCCW_KSM09.docx `
  --out data/pilot/NRCCW_KSM09 --languages sw --max-segments 60

# Full run (all segments, Swahili / English / auto language) + review chunks:
.\.venv\Scripts\python.exe -m transcribe.pilot.run `
  --audio "data/audio/NRCCW_ KSM09.MP3" `
  --transcript data/audio/NRCCW_KSM09.docx `
  --out data/pilot/NRCCW_KSM09 --export-review
```

The first run downloads the MMS-300m aligner (about 1.2 GB) and Whisper small (about
0.5 GB). Measured on a 4-core CPU: alignment emissions run at about 0.2x real time (one hour
of audio in roughly 12 to 15 minutes, cached afterwards) and Whisper small at 0.3 to 0.6x real
time per language setting. Add `--models small,large-v3-turbo` to compare the larger model (slower on CPU).
Outputs in `data/pilot/NRCCW_KSM09/`:

| File | Contents |
|---|---|
| `report.md`, `report.json` | Alignment quality, lowest-confidence turns, segment stats, WER/CER, per-language WER, switch-point errors, words emitted in untranscribed gaps, worst segments |
| `segments.json` | Aligned training-sized segments (2 to 30 s) with flags |
| `review_manifest.json`, `chunks/` | Pre-filled chunks for the review UI (with `--export-review`) |
| `alignment.json`, `emissions.npy`, `asr_*.json` | Caches |

Review the pre-filled chunks and measure your speed:

```powershell
.\.venv\Scripts\python.exe -m transcribe.app --open data/pilot/NRCCW_KSM09/review_manifest.json
# after reviewing (the app prints the session id):
.\.venv\Scripts\python.exe -m transcribe.pilot.review_stats data/app/sessions/<session id>/session.json
```

The reports contain transcript text and numbers, not audio.

## Pilot on Kaggle (GPU, no monitoring)

1. Kaggle > **Datasets > New Dataset** (private): upload the audio files and their DOCX
   transcripts. Names must match (`NRCCW_ KSM09.MP3` + `NRCCW_KSM09.docx`; case, spaces and
   underscores are ignored).
2. Kaggle > **New Notebook** > File > Import Notebook > this repo's
   `notebooks/kaggle_pilot.ipynb` (or paste its code cell). **Add Data** > your dataset.
   **Settings**: Accelerator = GPU, Internet = On.
3. **Save Version > Save & Run All (Commit)** and close the tab.
4. When it finishes: open the version > **Output** > download `pilot_reports.zip`
   (`summary.md` across recordings, plus each recording's `report.md`).

The notebook's only code:

```
!rm -rf transcribe && git clone -q --depth 1 -b matt/relaxed-meitner-2lfrx6 https://github.com/mattobryan/transcribe.git
!cd transcribe && bash scripts/kaggle_pilot.sh
```

`scripts/kaggle_pilot.sh` installs dependencies, runs the pilot tests, checks that Whisper can use
the GPU (falls back to CPU if not), runs every recording it finds, and zips the reports. Options
go before `bash`: `MODELS=small LANGS=sw MAX_SEGMENTS=150 ONLY=NRCCW_KSM09 bash scripts/kaggle_pilot.sh`.

## Prepare Reviewed Long Recordings

The annotation workflow starts from the complete audio and reviewed DOCX transcript. It
keeps both source files unchanged, detects candidate speech regions, exports short WAV
chunks, and writes a review manifest.

```bash
python -m pip install -r requirements-annotation.txt
python -m transcribe.script.parse_transcript \
  --transcript data/transcripts/NRCCW_KSM09.docx \
  --audio data/audio/NRCCW_KSM09.wav \
  --output data/projects/NRCCW_KSM09/transcript.json
python -m transcribe.script.prepare_audio \
  --audio data/audio/NRCCW_KSM09.wav \
  --output-dir data/projects/NRCCW_KSM09/chunks \
  --manifest data/projects/NRCCW_KSM09/audio.json \
  --transcript-manifest data/projects/NRCCW_KSM09/transcript.json
python -m transcribe.app --open data/projects/NRCCW_KSM09/audio.json
```

After reviewing and adding approved transcripts to the rich segment manifest:

```bash
python -m transcribe.script.export_training_manifest \
  --input data/projects/NRCCW_KSM09/reviewed.json \
  --output data/metadata.json
python -m transcribe.legacy.train --config config/default.yaml
```

The first implementation uses energy-based VAD and manual review. Forced alignment can
be added later as an optional suggestion layer; it must not overwrite the reviewed text.

## Legacy configuration

Settings are in `config/default.yaml`. Key fields:

| Field             | Description                         | Default |
|-------------------|-------------------------------------|---------|
| `sample_rate`     | Audio sampling rate (Hz)            | 16000   |
| `n_mfcc`          | Number of MFCC coefficients         | 13      |
| `model_type`      | `ctc` (recommended) or `seq2seq`    | ctc     |
| `hidden_dim`      | LSTM hidden units                   | 256     |
| `num_layers`      | LSTM layers                         | 3       |
| `bidirectional`   | Bidirectional LSTM                  | true    |
| `learning_rate`   | Initial LR                          | 0.001   |
| `batch_size`      | Training batch size                 | 16      |
| `epochs`          | Number of training epochs           | 100     |
| `eval_every`      | Run validation every N epochs       | 5       |
| `save_every`      | Save checkpoint every N epochs      | 10      |
| `grad_clip`       | Max gradient norm                   | 1.0     |

## Legacy model architectures

### CTC Model (Recommended)

- CNN feature extractor (3 Conv1D layers, reduces sequence by 8x)
- Bidirectional LSTM encoder
- Per-frame logits with CTC loss
- Greedy decoding or beam search

### Seq2Seq Model

- CNN + BiLSTM encoder
- LSTM decoder with additive attention
- Teacher forcing during training (configurable ratio)
- Greedy decoding during inference

## Dependencies

Grouped in `requirements/` (and as extras in `pyproject.toml`): `core` (transcript parsing, audio
I/O), `pilot` (PyTorch, transformers, faster-whisper, jiwer), `ui` (FastAPI, Uvicorn), `legacy` (PyTorch,
scikit-learn, edit-distance metrics, NLTK, PyYAML) and `dev` (pytest, ruff).
