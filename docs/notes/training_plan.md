# Training plan: from the corrected recordings to a Kenyan Kiswahili-English model

Status 2026-10-05: the code is in `src/transcribe/train/`. The data pipeline and the scorer are unit-tested (`tests/train`).
The fine-tune was run end to end on CPU with `whisper-tiny` on synthetic clips (loss falls, dev score and adapter are written);
it has **not** been run on real data or on a GPU, so expect to adjust batch size and learning rate on the first Kaggle run.

Consent: the user confirmed that the three 3-hour recordings may be used for training. Keep recordings, transcripts, logs and
language labels out of git.

## What the corrector produces for each recording

1. `<name> corrections log.json`: every chunk with Scribe's first text, the page's mechanical clean-up and the final human text, and
   whether the chunk was checked.
2. `<name> language labels.csv`: every word of the final text with `sw` / `en` / `name` / `mixed`, the parts of mixed words
   (`ni:sw|break:en|ie:sw`) and where the label came from.
3. `<name>.docx`: the client deliverable (not used for training).

## Splits (decide once, never change)

A recording is never divided. With the four recordings: **test** = one 3-hour recording, fully corrected and never trained on;
**dev** = the first 64-minute interview (used to pick the epoch and settings); **train** = the other two 3-hour recordings.
More recordings go to train. The same interviewer is in every recording, so the report also gives WER for the interviewer (I)
and the respondent (R) separately.

## Steps

1. `python -m transcribe.train.prepare recordings.json --out data/train` (see the docstring for the file). Only checked chunks;
   the training text is the words in order with the project's dash and ellipsis marks removed.
2. `python -m transcribe.train.evaluate data/train/test.jsonl --systems scribe_raw scribe_cleaned hf:openai/whisper-small`: the
   baselines. The report gives WER, CER, WER by language (sw, en, mixed, name), the error rate at Swahili/English switch points and
   the I/R split. Scribe's score is flattering: the reference is Scribe's own text after a person edited it.
3. `python -m transcribe.train.finetune data/train --model openai/whisper-small --out checkpoints/run1` (LoRA; `--full` for all
   weights). Or run `notebooks/kaggle_finetune.ipynb`. Each chunk gets the language token of its main language; compare
   `--language dominant`, `auto` and `sw` at evaluation.
4. Score the adapter on the test split with `hf:<model>@<adapter dir>`. The model is better only if it beats `scribe_cleaned` or
   whisper-small zero-shot on the test recording, and especially on mixed words and switch points.

## Expectations (rules of thumb, not measured here)

- About 6 to 7 hours of corrected audio for training is enough to adapt Whisper to these speakers, topics and accents with LoRA.
  It will not make a general Kenyan model: that needs many more speakers (TV, church, street speech). The test recording is the
  honest measure of progress; add more recordings to train, never to test.
- Backchannels the other speaker says mid-sentence are not in the text but are in the audio, so there is some label noise.
- If the model does not beat Scribe on the test recording, use the corrections log as the list of what to fix next (word list,
  mixed-word rules) before training more.
- Check ElevenLabs' terms before training on bulk Scribe transcripts of audio nobody has corrected.
