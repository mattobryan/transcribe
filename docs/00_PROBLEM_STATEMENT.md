# Problem Statement

## The problem

Kenyans rarely speak "pure" English or "pure" Kiswahili. Everyday speech on TV and radio, in churches, in meetings, in interviews and on the street switches between the two:

- **Between sentences:** "Tulienda kanisa jana. The sermon was about patience."
- **Within a sentence:** "Nilikuwa nataka ku-apply lakini the deadline ilikuwa imepita."
- **Within a word:** an English stem inside Kiswahili grammar, e.g. *nime-download*, *ame-realize*, *tuta-organize*, *wanam-text*.

It also carries Kenyan English pronunciation, Sheng-influenced vocabulary, and accent effects from other first languages such as Dholuo, Luhya and Kikuyu.

Off-the-shelf speech-to-text systems serve this speech poorly:

1. **Monolingual systems** (English-only or Swahili-only) either drop the other language or force it into their own spelling (e.g. "download" becomes "daunlodi").
2. **Multilingual systems** such as stock Whisper pick one language per 30-second window. On mixed speech they tend to *translate* the minority language instead of transcribing it, and they hallucinate on noise, music and silence.
3. **None of them is tuned to Kenyan pronunciation.** Word error rates on Kenyan-accented, noisy, multi-speaker audio are far higher than their published benchmarks.

As a result, people who need Kenyan speech in text form (media houses subtitling shows, churches archiving sermons, researchers transcribing interviews, organisations minuting meetings) still transcribe by hand. That costs roughly 4–8 hours of human time per hour of audio.

## What we have

- A large private collection of Kenyan recordings with **human-reviewed transcripts** (DOCX, sometimes PDF), transcribed at the interview level.
- A partial pipeline in this repo: transcript parsers, energy-based VAD chunking, a Gradio review UI, and two from-scratch CNN-LSTM models (CTC and Seq2Seq) on MFCC features.
- CPU for development and inference, and a Kaggle account for GPU training.

## Why the current pipeline cannot reach the goal

| Gap | Consequence |
|---|---|
| VAD chunks are not linked to transcript text | Each chunk has to be typed from scratch, so the existing reviewed transcripts are wasted |
| From-scratch MFCC CNN-LSTM with 256 ms CTC frames | Cannot learn: most chunks have more target characters than output frames, and `zero_infinity=True` hides the resulting zero loss |
| Tokenizer removes spaces | Output is not readable text |
| Random chunk-level train/test split within one recording | Evaluation results are inflated by leakage |
| The Transcribe button overwrites the reviewer's text | Breaks the rule that reviewed text is authoritative |
| No inference service, no long-audio handling, no exports | Nothing a user outside the repo can use |

## Problem statement (one sentence)

> Kenyan speakers mix English and Kiswahili between sentences, within sentences and within words, and no accessible speech-to-text tool transcribes this faithfully. We will build a deployable transcriber, fine-tuned from a multilingual pretrained speech model on our own reviewed Kenyan recordings. It will produce readable, correctly spelled code-switched text from real-world audio (interviews, TV, church, meetings), and it will include a corpus workflow that turns existing human transcripts into training data with minimal manual effort.

## Success looks like

- A user uploads an hour of Kenyan audio on a CPU machine and gets back a readable, timestamped, editable transcript with English words spelled as English and Kiswahili as Kiswahili. The run finishes in under the audio's own duration.
- **Clean speech:** word error rate at or below 20% at v1 and at or below 10% as the stretch goal ("90% correct"). Noisy and overlapping audio is measured separately.
- A new reviewed recording goes from DOCX + audio to approved training segments in a fraction of real time, because text is pre-aligned and the reviewer only verifies it.
- Every model version is reproducible from a frozen dataset snapshot and a Kaggle training run.

## Out of scope (for now)

- Speaker identification, and any inference of ethnicity or tribe.
- Transcribing languages other than English and Kiswahili. Isolated Dholuo or other words are *tagged*, not modelled as a third target language.
- Translation.
- Real-time streaming (batch file transcription first).
- Multi-tenant SaaS, billing, and public accounts.
