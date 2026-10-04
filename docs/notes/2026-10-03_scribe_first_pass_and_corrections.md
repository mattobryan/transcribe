# Scribe as first pass, and what people still correct (first interview)

Recording: one 64-minute interview (nurse, postnatal ward), English with Kiswahili mixed in, two speakers.
Three transcripts of it were compared: a small Whisper run on CPU (language fixed to Swahili), a second tool's output,
and ElevenLabs Scribe (auto language). Only the first and third are ours to judge here.

## What we saw

- Whisper small / CPU / `sw`: 164 chunks, about 15 chunks that were pure repetition loops ("kwa kwa kwa ...") and about 20
  more with garbled Kiswahili. English was readable. Fixed in code by a loop guard (retry with a repetition penalty,
  collapse what is left, flag the chunk), but not good enough to be the main transcript.
- Scribe: no loops, real Kiswahili words, correct names such as the hospital. It is a better first pass.
- What a person still fixes after Scribe (318 chunks):
  1. Diarization of very short turns ("it's okay" given to the wrong speaker) and where a backchannel belongs.
  2. Words written as one piece that are really two languages: `anafeel`, `unadeal`, `wanakoconsult`, `ukfeel`,
     `tunabidi`, `tutamisse`, `zinezinamotivate`, `mnitaita`. The Swahili prefix must be tagged Swahili and the English
     stem left English (`ni` + `break` + `ie`).
  3. Misheard words: `locummers` (locums), `aeration`, `rusa`, `asongee`, `soile`, `wueh`, `mem`.
  4. Project conventions Scribe does not apply: one dash = cut off by the other speaker, two dashes = self-correction,
     ellipsis = the speaker's own pause; capitals where pieces join; a comma after a sentence-opening "So".
- Scribe's own confidence was a weak signal: only 13 words below logprob -1, so "flag low confidence" caught 3 chunks.

## Rules now in the corrector (so people do not redo them by hand)

Remove a speaker's own fillers; keep the other speaker's "Mm-hmm" only at the end of a sentence and ignore it mid-sentence;
strip dash + ellipsis when only a backchannel interrupted; join pieces with the right capital letter; "So," comma; italic
Swahili by word list plus verb-prefix rules (`ku`, `tuna`, `ana` + English stem); flags for the cases it cannot decide.
First pass on this interview: 119 inline fillers removed, 363 backchannel lines kept, 165 mid-sentence ones dropped, 75 "So,"
commas, 161 dashes and 158 ellipses removed at joins.

## Decision for the main project: do not rely on Scribe alone

Use Scribe (or another strong model) as the first-pass labeller, not as the product.

- Why not rely on it: it is an outside paid service (cost per hour, availability, consent and retention for research
  audio), we cannot adapt it to Kenyan speech, and it still makes the mixed-word and speaker errors above.
- What to rely on it for: speed. A person correcting Scribe is far faster than transcribing from scratch.
- What we keep: audio chunk + Scribe text + the corrected text + the Swahili/English tags a person adds. That triple is the
  fine-tuning set for our own model (Whisper large-v3 / turbo) and the frozen test set to score it, with Scribe's raw output
  as the baseline to beat. The corrections log from the corrector (`*.corrections log.json`) is exactly that record.
- Mixed words need our own handling either way: the language tags people add (`_ni_break_ie_`) are the training signal for a
  word-level language label; the Swahili word list is only a stopgap.

## Next

1. Finish correcting this interview; export the corrections log and the word-changes CSV.
2. Count the error classes above from the log (how many mixed words, how many misheard, how many diarization fixes).
3. Repeat on two or three more recordings with different speakers and accents before choosing a model size.
4. Fine-tune on the corrected chunks (grouped by speaker/recording, never split a recording across train and test) and
   compare WER against Scribe on the same frozen test chunks.
