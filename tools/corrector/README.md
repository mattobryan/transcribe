# Transcript Corrector

A single-page tool (no server) for correcting an ElevenLabs Scribe transcript against its recording, built for
Kenyan English / Kiswahili interviews. It runs in the browser; nothing is uploaded.

- `index.html` is the built page (what is published as the artifact). Rebuild with `python build.py` after editing `src/`.
- `src/` holds the parts (`head`, `style`, `body`, `script`) and the two word lists used for Swahili italics.
- `lexicon.py <scribe.json>` rebuilds `src/sw_words.txt` / `en_words.txt` (needs `wordfreq` and `lingua-language-detector`).
- Do not commit transcripts or recordings: the interviews are consented research data.

What it does: chunks the transcript by speaker and pause, labels turns `I:` / `R:`, removes a speaker's own fillers,
keeps backchannels only at the end of a sentence, repairs dashes, ellipses and capitals where chunks join, adds the comma
after a sentence-opening "So", italicises Swahili (and parts of mixed words), and exports a Word file (Times New Roman 12,
header, bold capital title, I:/R: key, "Page x of y") plus a **corrections log** (Scribe's first text, the automatic
clean-up, and the final human text, with word-level changes and the Swahili/English labels the person added).
