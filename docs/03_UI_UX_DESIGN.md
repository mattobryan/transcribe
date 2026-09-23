# UI/UX Design

| | |
|---|---|
| Status | Draft v1 |
| Stack | Gradio Blocks mounted inside FastAPI (v1). Custom React front end only if Gradio blocks a P0 requirement |
| Related | [01 PRD](01_PRD.md), [04 App Flow](04_APP_FLOW.md) |

## 1. Design principles

1. **Verify, don't type.** Every text box opens pre-filled (aligned transcript or model hypothesis). The fastest action is "approve".
2. **Keyboard first.** A reviewer should process a segment without touching the mouse.
3. **The model suggests, the human decides.** Model output is always visually distinct (a diff panel) and is never silently written into the editable text.
4. **One primary action per screen**, shown with the primary colour. Destructive actions are secondary and need confirmation.
5. **State is always visible.** The segment's status, whether there are unsaved changes, job progress and the active model are shown on screen.

## 2. Visual language

This keeps the earth-tone theme already chosen in `review/app.py` and formalises it as tokens.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#f4eee5` | `#211b17` | page |
| `--surface` | `#fffaf3` | `#30261f` | cards, inputs |
| `--text` | `#2d241d` | `#f4eee5` | body |
| `--muted` | `#675443` | `#d5bda5` | secondary text, timestamps |
| `--border` | `#b99b7a` | `#806149` | dividers |
| `--accent` | `#8a5a3b` | `#b77b51` | primary buttons, focus ring |
| `--accent-hover` | `#6f432d` | `#d19669` | |
| `--ok` | `#4f7a3a` | `#8fbf6f` | approved |
| `--warn` | `#b07d1a` | `#e0b04f` | flagged, low confidence |
| `--danger` | `#9c3b2e` | `#e07a6a` | rejected, errors |
| `--lang-en` | `#2f5d8a` underline | `#7fb0e0` | language span tint |
| `--lang-sw` | `#4f7a3a` underline | `#8fbf6f` | |
| `--lang-mixed` | `#8a3b7a` underline | `#d08ac0` | |

Typography: Times New Roman (the project's choice) for UI text at 12 pt minimum. The transcript editor uses **14 pt** with a line height of 1.6, because reviewers read it for hours. Timestamps and IDs use a monospace font at 11 pt. Colour is never the only signal: flags also show an icon and a label.

## 3. Information architecture

```text
Top bar:  [Transcribe] [Corpus Studio] [Models & Evaluation]      active model: whisper-small-kct-v1 ●
          |
          |-- Transcribe
          |     |-- New transcription (upload)
          |     |-- Jobs list
          |     `-- Transcript editor
          |-- Corpus Studio
          |     |-- Recordings list
          |     |-- Import recording (wizard)
          |     |-- Recording overview (alignment report)
          |     |-- Review (segment reviewer)   <- most-used screen
          |     `-- Datasets (build / export snapshots)
          `-- Models & Evaluation
                |-- Model registry
                |-- Evaluation report
                `-- Compare two models
```

## 4. Screens

### 4.1 Transcribe: New transcription

```text
+------------------------------------------------------------------+
|  Drop audio or video here, or [Browse]                           |
|  WAV MP3 M4A MP4 OGG FLAC · up to 3 h                            |
+------------------------------------------------------------------+
|  Title  [____________________]   Domain [Interview v]            |
|  Model  [whisper-small-kct-v1 (production) v]                    |
|  [ ] Allow corrections to be added to training data (consent)    |
|                                            [ Start transcription ]|
+------------------------------------------------------------------+
```

- The consent checkbox is off by default and maps to `consent_scope`.
- After submitting, the user lands on the Jobs list with a progress bar: "Transcribing 12:40 / 58:10 · ETA 18 min".

### 4.2 Transcribe: Transcript editor

```text
+------------------------------------------------------------------+
| Sermon_2026-09-14.mp3 · 58:10 · whisper-small-kct-v1   [Export v]|
+------------------------------------------------------------------+
| ▶ ||  00:12:41 ──────────●──────────────────── 00:58:10   1.0x v |
+----------------------+-------------------------------------------+
| 00:12:31  S?         | Leo tunataka ku-focus kwa neno la Mungu,  |
| 00:12:38  S?         | and I want you to open your Bibles        |
|>00:12:41             | pale Mathayo sura ya tano. [edited]       |
| 00:12:49             | Wale wa nyuma mnasikia? ░low confidence░ |
+----------------------+-------------------------------------------+
| Enter: play segment · Ctrl+S: save · Alt+Up/Down: prev/next      |
| Ctrl+Shift+S: split at cursor · Ctrl+M: merge with next          |
+------------------------------------------------------------------+
```

- Clicking a segment seeks the player there, and the current segment is highlighted during playback.
- Low-confidence words (word probability < 0.5) get a dotted `--warn` underline, with the probability shown in a tooltip.
- Edited segments show an `[edited]` chip. Edits autosave after 1.5 s idle.
- The Export menu offers TXT, SRT, VTT, DOCX, JSON, plus "Promote edits to corpus" when consent allows.

### 4.3 Corpus Studio: Import recording (3-step wizard)

1. **Files:** audio (required), transcript DOCX/PDF (optional).
2. **Metadata:** title, domain, recorded date, region (free text, optional), consent scope (required: *training* / *eval only* / *transcription only*), and the language-hint convention for the DOCX ("italic = Kiswahili" ✓).
3. **Confirm:** a parsed preview showing the first 10 turns with speaker labels, the count of turns, the count of `[unclear]` markers, and the audio duration. Pressing **Import & align** enqueues the job.

### 4.4 Corpus Studio: Recording overview

```text
KSM09 · interview · 1:12:04 · consent: training
Alignment: done · mean confidence 0.81 · 612 segments
[████████████░░░░░░░] 402 approved · 31 flagged · 179 to review
Low-confidence turns (likely non-verbatim):  p0142 0.31 | p0388 0.28 | ...
Confidence timeline:  ▇▇▇▆▇▇▂▇▇▇▇▇▁▇▇▇▇▆▇  (click to jump)
                                          [ Start reviewing ▶ ]
```

The confidence timeline shows problem regions (non-verbatim passages, music, crosstalk) at a glance.

### 4.5 Corpus Studio: Segment reviewer (primary screen)

```text
+------------------------------------------------------------------+
| KSM09 · segment 143/612 · priority 0.71 · status: candidate      |
| Queue: [Priority v]  Filter: [All flags v]      unsaved ●        |
+------------------------------------------------------------------+
| waveform ▁▂▅▇▆▃▂▁▂▅▇▇▅▃▁▁▂▃▅▆▅▃▂▁   00:14:02.35 → 00:14:13.90     |
|   [◀◀ -250ms] [◀ -50ms] start [+50ms ▶]   end [◀ -50ms] [+50ms ▶]|
|   ▶ Play (Tab)   ↻ Replay last 3s (Shift+Tab)   0.75x / 1x       |
+------------------------------------------------------------------+
| Transcript (editable)                                     14 pt  |
| ┌──────────────────────────────────────────────────────────────┐ |
| │ Sasa mimi nilikuwa nimeapply hiyo job lakini they never      │ |
| │ called back, so nikaamua tu kuanza biashara yangu.           │ |
| └──────────────────────────────────────────────────────────────┘ |
| Source: aligned transcript p0141–p0142 · word conf ░0.42░ "nimeapply"|
+------------------------------------------------------------------+
| Model hypothesis (whisper-small-kct-v1)        [Run model (R)]   |
|  Sasa mimi nilikuwa nimeapply hiyo job lakini they never         |
|  called back so ~~nikaamua~~ **nikamua** tu kuanza biashara yangu|
|  WER vs transcript: 7.1%   [Use hypothesis text (Ctrl+H)]        |
+------------------------------------------------------------------+
| Flags: [U]nclear [O]verlap [N]oisy [M]usic [V]non-verbatim       |
| Language spans (select text, then E / S / X mixed / L luo)       |
| Speaker: interviewer ▾                                           |
+------------------------------------------------------------------+
| [Reject (Ctrl+Del)]  [Save draft (Ctrl+S)]  [Approve & next ⏎]   |
| History ▾  r3 you 14:02 · r2 aligner · r1 aligner                |
+------------------------------------------------------------------+
```

Behaviour:

- **Approve & next** (Ctrl+Enter) saves a revision with `status=approved` and loads the next item in the queue. Tab plays the segment. The segment auto-plays on load (a setting, on by default).
- Changing a boundary re-renders the slice and replays the last 3 s (for the end boundary) or the first 3 s (for the start boundary).
- **Use hypothesis text** is the only way hypothesis text enters the editor. It is explicit, and it creates a revision with `source=model_hypothesis`.
- Flags are toggles, with the key shown in brackets. Setting *Unclear* or *Non-verbatim* shows an inline note: "Excluded from training until resolved."
- Pressing Approve on a segment that has an active exclusion flag shows a confirmation: "Approve for eval only?"
- The History drawer lists revisions and offers "Restore r2" (which creates r4 as a copy of r2, so nothing is ever deleted).
- A 409 conflict shows a banner: "This segment changed in another tab. Reload / Keep mine as new revision."

Keyboard map:

| Key | Action |
|---|---|
| Tab / Shift+Tab | Play segment / replay last 3 s |
| Ctrl+Enter | Approve & next |
| Ctrl+S | Save draft |
| Ctrl+Del | Reject |
| Alt+↑ / Alt+↓ | Previous / next in queue |
| `[` `]` | Start −50 ms / +50 ms |
| `{` `}` | End −50 ms / +50 ms |
| Alt+U/O/N/M/V | Toggle flags |
| Alt+E/S/X/L | Tag selected text as en / sw / mixed / luo |
| R | Run the model on the segment |
| Ctrl+H | Copy hypothesis into the editor |

The shortcuts are bound with a small injected JS snippet (`gr.Blocks(js=...)`), because Gradio has no native global hotkeys.

### 4.6 Corpus Studio: Datasets

- **Filter builder:** recordings (multi-select), domains, consent = training (locked), status = approved, exclude flags (default: unclear, non_verbatim, overlap), speaker roles (default: all), min/max duration.
- **Split:** "Use frozen test set v1" is locked. Dev % and seed are settable. A preview table shows each split's hours, segments, recordings and code-switch density, with a leakage check ✓.
- **Build snapshot** creates `kct-ds-v3 (frozen)`. **Export for Kaggle** produces a zip of the folder plus a README with the dataset hash.

### 4.7 Models & Evaluation

- **Registry table:** name, base model, dataset snapshot, dev WER, test WER, status (candidate/production/retired), and actions (Evaluate, Promote, Convert to CPU format).
- **Evaluation report:** metric cards (WER, CER, switch-point, hallucination/min, RTF), a breakdown table by domain × flag, and the worst 50 segments with audio, reference/hypothesis diff, and a "Send to review" action.
- **Compare:** two models on the same split, showing the per-metric delta, segments where A beats B and vice versa, and error categories (substitution en→sw and sw→en, deletions at switch points).

## 5. States and feedback

| State | Treatment |
|---|---|
| Job queued/running | Progress bar with processed/total seconds and ETA; the page can be closed |
| Job failed | Red card with the error summary, a "Retry" action and the log link |
| No production model | Transcribe is disabled with the notice: "Register a model in Models & Evaluation"; Review still works without "Run model" |
| Empty review queue | "All segments reviewed 🎉. Build a dataset snapshot →" |
| Unsaved changes | A dot in the header, and a browser `beforeunload` warning |
| Offline aligner model missing | A setup checklist on first run (ffmpeg, aligner weights, VAD weights) |

## 6. Accessibility

- Every action is keyboard-accessible, and the focus ring is a 2 px `--accent` outline.
- Contrast is at least 4.5:1 for text in both themes (the tokens above meet this for body text on surface).
- Audio controls have labels; screen-reader text is set on the flag toggles.
- Playback speed runs from 0.5× to 1.5× to help with fast or accented speech.

## 7. Gradio feasibility notes

| Need | Approach |
|---|---|
| Waveform + boundary nudging | `gr.Audio` waveform for playback; the boundaries are numeric fields with nudge buttons (slices are re-rendered server-side). No drag handles in v1 |
| Diff rendering | `gr.HTML` with a server-side word diff (`difflib` on normalised tokens) |
| Language span tagging | `gr.HighlightedText` for display; tagging through selection + hotkey handled by JS, which posts `(start, end, lang)` |
| Global hotkeys | `gr.Blocks(js=...)` keydown listener that clicks hidden buttons |
| Multi-page | `gr.Tabs` for the three surfaces, with FastAPI serving the downloads |

If boundary dragging or span tagging proves too clumsy in Gradio after M1, those two widgets move to a small custom component (`gradio cc`) rather than a full front-end rewrite.
