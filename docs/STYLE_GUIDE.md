# Transcription Style Guide

| | |
|---|---|
| Status | v0.1: decisions confirmed; examples to be extended in Phase 0 (P0.8) |
| Applies to | Source transcripts (DOCX), reviewer edits in Corpus Studio, and training targets |

## 1. Two layers of text

| Layer | Where | Numbers | Purpose |
|---|---|---|---|
| **Source text** | The DOCX transcripts | Digits (`6`, `2019`, `Level 5`) | Easy for people to write and read. Never modified by the pipeline |
| **Spoken form** | Segment revisions → training targets | Words in the language actually spoken (`sita` or `six`, `twenty nineteen`, `Level Five`) | What the model hears and learns |
| **Display form** | Transcribe output (optional) | Digits again, via inverse text normalisation | Readable subtitles and documents |

The pipeline converts source text to spoken form automatically (§4). You keep writing digits in the documents.

## 2. Language marking in DOCX

- **Italic = Kiswahili**; roman (upright) = English. This applies to every document.
- **Hybrid words:** write them as **one word, no hyphen**. Italicise only the Kiswahili part if you can: *nime*download, *wame*organize, *tuta*text*iana*. The parser tags a word that is part italic and part roman as `mixed`. If splitting the formatting is too slow, italicising the whole hybrid word is acceptable; the reviewer can retag it.
- Sheng words written with Kiswahili grammar are italic. English loanwords already absorbed into Kiswahili spelling (*kompyuta*, *basi*) are italic.
- Words that are neither English nor Kiswahili (e.g. Dholuo): write them as heard and wrap them in `[luo: …]`, e.g. `[luo: ber ahinya]`.

## 3. Verbatim rules

- Transcribe what was said, including repetitions that carry meaning and every code-switch.
- Drop pure fillers (*eh*, *umm*, *aah*) unless they carry meaning (*eeh* = "yes").
- Unintelligible speech: `[unclear]`. Do not guess. These segments are excluded from training until resolved.
- Crosstalk: `[overlap]` at the start of the overlapping part.
- Passages you paraphrased or skipped for meaning: `[nv]` … `[/nv]` (non-verbatim). These segments are excluded from training until corrected in review.

## 4. Numbers

Source documents always use **digits**. The spoken form uses **words in the language actually spoken**.

### 4.1 Default language rules (applied automatically)

| Case | Rule | Source → spoken |
|---|---|---|
| Number inside an *italic* (Kiswahili) span | Kiswahili words | *watu 6* → watu sita |
| Number inside a roman (English) span | English words | 6 people → six people |
| Year (4 digits, 1900–2099) | English words, year style | 2019 → twenty nineteen; 2005 → two thousand and five |
| Label + number (Level, Grade, Form, Class, Chapter, Room, Route, Phase…) | English words, title case | Level 5 → Level Five; Form 4 → Form Four |
| Money | Follow the span language; the currency word stays as spoken | *shilingi 500* → shilingi mia tano; 500 bob → five hundred bob |
| Phone and ID numbers | Digit by digit, span language | 0712 → zero seven one two |
| Time | Follow the span language | *saa 3* → saa tatu; 3 pm → three pm |

### 4.2 Why a rule alone isn't enough, and how it gets resolved

A bare `6` can't tell you whether the speaker said *sita* or *six*. Kenyans often say numbers, and especially years and money, in English inside Kiswahili sentences. So the pipeline doesn't trust the rule blindly:

1. It generates both candidates (`sita` and `six`; `twenty nineteen` and `elfu mbili kumi na tisa`).
2. The forced aligner scores each candidate against the audio, and the higher-scoring one wins.
3. If the two scores are too close, the segment gets a `number_check` flag and the reviewer hears it and picks.

In practice the audio decides, and the §4.1 table is only the tie-breaker.

### 4.3 Kiswahili number words (reference)

1 moja · 2 mbili · 3 tatu · 4 nne · 5 tano · 6 sita · 7 saba · 8 nane · 9 tisa · 10 kumi · 11 kumi na moja · 20 ishirini · 30 thelathini · 40 arobaini · 50 hamsini · 60 sitini · 70 sabini · 80 themanini · 90 tisini · 100 mia moja · 250 mia mbili hamsini · 1,000 elfu moja · 2019 elfu mbili kumi na tisa · 1,000,000 milioni moja

Noun-class agreement (*watu watatu*, *vitu vitatu*) is written **as spoken** by the reviewer. The automatic converter emits the bare counting form (*tatu*), and the aligner or reviewer corrects it to the agreeing form.

## 5. Other spelling conventions

| Item | Convention |
|---|---|
| Casing | Sentence case; proper nouns capitalised in both languages |
| Punctuation | Normal punctuation in transcripts (Whisper learns it); ignored when scoring |
| Contractions | As spoken: *don't*, *I'm*; Kiswahili elisions as spoken (*nshaenda*) |
| Okay | *okay* (not *ok*, *OK*, *sawa* unless *sawa* was said) |
| Acronyms | Uppercase, no dots: *KCSE*, *NHIF*; letter-by-letter acronyms stay uppercase |
