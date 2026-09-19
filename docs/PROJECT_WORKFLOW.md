# Kenyan Speech Transcription Workflow

## Purpose

This project is an English-Kiswahili code-switching speech transcriber focused on Kenyan speech. The current direction is to build a carefully reviewed speech corpus that preserves Kenyan English pronunciation, Kiswahili, code-switching, local vocabulary, dialectal variation, and natural speech patterns.

The immediate goal is not speaker identification. The primary training record is:

```text
audio chunk -> accurate spoken transcript
```

Speaker, overlap, and unclear-speech information remain optional metadata for quality control.

## Settled Processing Plan

```text
Complete raw audio
        +
Reviewed transcript PDF/DOCX
        |
        v
Transcript parsing
        |
        v
Voice activity detection (VAD)
        |
        v
Natural-pause candidate chunks, maximum 30 seconds
        |
        v
Per-chunk review UI
        |
        +--> optional model transcription hypothesis
        |
        v
Human-corrected transcript and WER
        |
        v
Approved training manifest
        |
        v
Existing CTC/Seq2Seq training pipeline
```

The reviewed transcript is authoritative for wording. Audio review confirms what was actually spoken and supplies the chunk boundaries. Automatic alignment or transcription is a suggestion only; it must never silently replace the reviewed text.

## Implemented Changes

### Transcript ingestion

Added `src/transcribe/preprocessing/` with:

- `schema.py`: versioned `Manifest`, `TranscriptTurn`, and `TextSpan` records.
- `docx_parser.py`: reads paragraph order, `I:`/`R:`/`P1` speaker markers, run-level italics, annotations, and transcript text.
- `pdf_parser.py`: reads text-based reviewed PDFs such as `NRCCW_KSM09.pdf` and creates ordered transcript turns. Plain PDF extraction cannot reliably preserve italic formatting, so PDF language hints are stored as `unknown`.
- `manifest.py`: reads/writes rich manifests and exports approved records to the existing two-field training format.

The DOCX path can preserve italic/normal language hints. The PDF path preserves the transcript text and ordering but requires later manual language review if language spans are needed.

### Audio preparation

- `vad.py`: deterministic energy-based speech activity detection using Librosa RMS energy.
- `chunker.py`: exports WAV chunks with source start/end offsets, duration, sequence, VAD metadata, and candidate status.
- `prepare_audio.py`: CLI for VAD and candidate chunk generation.

Current real-recording result:

- Recording: `NRCCW_KSM09`
- Source audio: `data/audio/NRCCW_ KSM09.MP3`
- Source transcript: `data/audio/NRCCW_KSM09.pdf`
- Parsed transcript turns: `1,237`
- Detected speech intervals: `273`
- Candidate WAV chunks: `429`
- Maximum candidate chunk duration: `30` seconds

### Review interface

Added `src/transcribe/review/app.py` and `src/transcribe/script/review.py`.

The visible workflow is intentionally simple:

1. Chunk number and total chunk count.
2. Audio chunk player.
3. `Transcribe` button.
4. Editable transcript field.
5. `Previous chunk` and `Next chunk`.
6. Full reviewed script reference.
7. `Save and Continue`.
8. `Save and Exit`.

The UI also keeps technical information in the manifest, including WER, VAD data, timestamps, model hypothesis, source transcript ID, and optional speech flags. The chunk metadata is available in a collapsible technical panel.

The interface uses a light cream background, brown/earth-tone controls, Times New Roman, and 12pt text.

### Model transcription and WER

If a checkpoint exists, the UI automatically searches for:

```text
checkpoints/final_model.pt
checkpoints/best_model.pt
```

The user does not need to enter a checkpoint. The `Transcribe` action loads the matching vocabulary beside the checkpoint, extracts MFCCs from the current chunk, runs the saved CTC or Seq2Seq model, and records the model hypothesis.

WER is calculated per chunk by comparing:

```text
model hypothesis vs human-corrected transcript
```

If no checkpoint exists, the UI remains usable for manual transcription and reports that model transcription is unavailable. There is currently no trained checkpoint in `checkpoints/`.

### Training integration

- `export_training_manifest.py` projects approved rich segments into:

```json
[
  {
    "audio_file": "chunks/example_000001.wav",
    "transcript": "exact spoken text"
  }
]
```

- `src/transcribe/data/preprocessing.py` now accepts absolute generated chunk paths as well as paths under the configured audio directory.
- Existing model and trainer code remains unchanged.
- Approved segments can therefore feed the current training pipeline without discarding the richer review metadata.

### Project agent

The workspace custom agent at `.github/agents/transcription-workflow.agent.md` captures the
same architecture and review rules for future coding sessions. It explains the existing
CTC/Seq2Seq pipeline, the preprocessing boundary, checkpoint contract, validation method,
and long-audio workflow.

## Generated Project Files

The current NRCCW project is stored under:

```text
data/projects/NRCCW_KSM09/
|-- transcript.json   # parsed reviewed PDF turns
|-- audio.json        # VAD/chunk/review manifest
`-- chunks/           # generated candidate WAV clips
```

Original audio and transcript sources remain under `data/audio/` and are not modified.

## Commands

Use the project virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-annotation.txt
```

Parse the reviewed transcript:

```powershell
.\.venv\Scripts\python.exe -m src.transcribe.script.parse_transcript `
  --transcript data/audio/NRCCW_KSM09.pdf `
  --audio "data/audio/NRCCW_ KSM09.MP3" `
  --output data/projects/NRCCW_KSM09/transcript.json
```

Prepare speech intervals and chunks:

```powershell
.\.venv\Scripts\python.exe -m src.transcribe.script.prepare_audio `
  --audio "data/audio/NRCCW_ KSM09.MP3" `
  --output-dir data/projects/NRCCW_KSM09/chunks `
  --manifest data/projects/NRCCW_KSM09/audio.json `
  --transcript-manifest data/projects/NRCCW_KSM09/transcript.json `
  --recording-id NRCCW_KSM09 `
  --max-seconds 30
```

Launch the review UI:

```powershell
.\.venv\Scripts\python.exe -m src.transcribe.script.review `
  --manifest data/projects/NRCCW_KSM09/audio.json
```

Open:

```text
http://127.0.0.1:7860
```

After approving reviewed chunks, export training metadata:

```powershell
.\.venv\Scripts\python.exe -m src.transcribe.script.export_training_manifest `
  --input data/projects/NRCCW_KSM09/audio.json `
  --output data/metadata.json
```

Then use the existing trainer:

```powershell
.\.venv\Scripts\python.exe -m src.transcribe.script.train `
  --config config/default.yaml
```

## Per-Chunk Review Rules

For each chunk:

1. Listen to the complete chunk.
2. Click `Transcribe` when a checkpoint is available.
3. Correct the editable transcript to match the audio exactly.
4. Preserve Kenyan English, Kiswahili, code-switching, hesitations, and natural wording.
5. Do not guess unclear speech. Mark or leave it for later review.
6. Use `Save and Continue` to approve/save progress and move forward.
7. Use `Save and Exit` to preserve the current chunk and resume later.

A model hypothesis is not ground truth. The corrected transcript is the training target.

## Plan Status

### Complete

- [x] Preserve complete raw audio and reviewed transcript sources.
- [x] Parse reviewed PDF transcripts into ordered turns.
- [x] Add DOCX parsing with italic/normal run metadata.
- [x] Add versioned rich manifests.
- [x] Add energy-based VAD.
- [x] Add natural-pause candidate chunk export.
- [x] Add per-chunk review persistence and resume state.
- [x] Add simple chunk navigation and save workflow.
- [x] Add automatic checkpoint discovery.
- [x] Add optional model transcription and per-chunk WER storage.
- [x] Add approved-manifest export for the existing trainer.
- [x] Add generated-path support to dataset preparation.
- [x] Validate parser, VAD/chunking, review persistence, WER, export, and Python compilation.

### Next

- [ ] Train an initial checkpoint from approved chunks.
- [ ] Use that checkpoint through the automatic `Transcribe` button.
- [ ] Add optional forced-alignment suggestions for word-level timings.
- [ ] Add richer visual waveform/timeline editing only if segment-level review needs it.
- [ ] Add grouped train/validation/test splitting by recording and speaker/session where metadata is available.
- [ ] Add long-recording reconstruction and full-recording WER/CER evaluation.
- [ ] Add active-learning reports that prioritize high-WER or uncertain chunks.
- [ ] Decide separately whether to preserve spaces and remove `<SOS>/<EOS>` from CTC targets in a vocabulary/checkpoint migration.

## Terminology

The appropriate learning loop is:

```text
unlabeled pretraining, if used
+ supervised fine-tuning
+ human-in-the-loop correction
+ active learning
```

This is not reinforcement learning unless a reward-based environment is introduced. For this speech corpus, corrected transcripts and WER-driven review are the useful feedback signal.

## Known Limitations

- The current PDF parser cannot recover italic formatting from plain extracted PDF text.
- Energy VAD proposes speech regions; it does not understand words or guarantee perfect boundaries.
- The current candidate chunks are not automatically aligned to exact transcript words.
- The current model checkpoint directory is empty, so `Transcribe` will report that no model is available until training produces one.
- Existing model tokenization removes spaces from targets; changing that requires a deliberate vocabulary/checkpoint migration.
- The current data split logic should eventually be changed to recording/session-level grouping to prevent leakage.
- Automatic speaker or tribal identity inference is intentionally out of scope.
