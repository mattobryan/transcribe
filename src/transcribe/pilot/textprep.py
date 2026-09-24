"""Turn reviewed transcript turns into words ready for alignment and scoring.

Each word keeps its original spelling (the label) plus two derived forms:
``align`` (letters the aligner can score) and ``norm`` (the scoring form).
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

# Transcriber markers: "[Unclear 13:10-13:15]", "[No speech - activity, 17:22-17:24]",
# and the looser "(Unclear: 1:16:07-1:16:08]]" forms seen in real transcripts.
ANNOTATION = re.compile(
    r"\[+[^\]]*\]+"
    r"|\((?:unclear|no speech|inaudible|crosstalk|overlap|laugh)[^)\]]*[)\]]+",
    re.IGNORECASE)
WORD = re.compile(r"\S+")
PARTICIPANT = re.compile(r"^P\d+[.,?!:;]*$")
BREAK = re.compile(r"-{2,}|\u2014|\u2026")


@dataclass
class Word:
    index: int
    turn_index: int
    speaker: Optional[str]
    raw: str
    norm: str
    align: str
    lang: str = "unknown"          # en | sw | mixed | unknown
    annotations: List[str] = field(default_factory=list)
    # Timestamped markers next to this word, e.g. "[Unclear 13:10-13:15]":
    # {"start": s, "end": s, "side": "before" | "after"}. Used to check alignment.
    markers: List[Dict] = field(default_factory=list)
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None

    @property
    def has_digit(self) -> bool:
        return any(ch.isdigit() for ch in self.raw)


def _strip_marks(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_word(raw: str) -> str:
    """Scoring form: lowercase, no punctuation or hyphens, inner apostrophes kept."""
    text = unicodedata.normalize("NFKC", raw).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w']", "", text).replace("_", "")
    return text.strip("'")


def normalize_text(text: str) -> List[str]:
    text = ANNOTATION.sub(" ", text)
    return [word for word in (normalize_word(token) for token in text.split()) if word]


def align_form(raw: str, alphabet: Optional[Iterable[str]] = None) -> str:
    """Letters the CTC aligner can score. Digits are dropped (interpolated later)."""
    text = _strip_marks(normalize_word(raw))
    text = "".join(ch for ch in text if ch.isalpha() or ch == "'")
    if alphabet is not None:
        allowed = set(alphabet)
        text = "".join(ch for ch in text if ch in allowed)
    return text


TIMESTAMP = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})")


def marker_times(marker: str) -> Optional[Dict]:
    """'[Unclear 1:05:21-1:05:24]' -> {'start': 3921.0, 'end': 3924.0}."""
    stamps = [int(h or 0) * 3600 + int(m) * 60 + int(s) for h, m, s in TIMESTAMP.findall(marker)]
    if not stamps:
        return None
    return {"start": float(stamps[0]), "end": float(stamps[-1])}


def annotation_kind(marker: str) -> str:
    lowered = marker.lower()
    if "unclear" in lowered or "inaudible" in lowered:
        return "unclear"
    if "overlap" in lowered or "crosstalk" in lowered or "cross talk" in lowered:
        return "overlap"
    if "no speech" in lowered:
        return "no_speech"
    return "other"


def select_turns(turns: Sequence[Dict]) -> List[Dict]:
    """Drop header paragraphs before the first speaker-labelled turn.

    Unlabelled paragraphs after that are kept as continuations of the previous
    speaker's turn.
    """
    selected = []
    speaker = None
    for turn in turns:
        if turn.get("speaker_id"):
            speaker = turn["speaker_id"]
        elif speaker is None:
            continue
        selected.append({**turn, "speaker_id": turn.get("speaker_id") or speaker})
    return selected


def _word_lang(text: str, start: int, end: int, spans: Sequence[Dict]) -> str:
    hints = set()
    for offset in range(start, end):
        if not text[offset].isalpha():
            continue
        for span in spans:
            if span["start_char"] <= offset < span["end_char"]:
                hints.add(span.get("language_hint", "unknown"))
                break
    hints.discard("unknown")
    if hints == {"sw"}:
        return "sw"
    if hints == {"en"}:
        return "en"
    if hints >= {"sw", "en"}:
        return "mixed"
    return "unknown"


def turns_to_words(turns: Sequence[Dict], alphabet: Optional[Iterable[str]] = None) -> List[Word]:
    words: List[Word] = []
    carried: List[Dict] = []        # timestamps from turns without words ("[No speech ...]")
    for turn_index, turn in enumerate(turns):
        text = turn.get("text", "")
        spans = turn.get("spans", [])
        markers = [(m.start(), m.end(), annotation_kind(m.group())) for m in ANNOTATION.finditer(text)]
        stamps = {m.start(): marker_times(m.group()) for m in ANNOTATION.finditer(text)}
        stamp_pending: List[Dict] = carried
        carried = []
        pending: List[str] = []
        prefix = ""
        turn_words: List[Word] = []
        masked = ANNOTATION.sub(lambda m: " " * len(m.group()), text)
        # False starts and cut-offs ("tuta--tutaanza", "tuna--,") are separate words.
        masked = BREAK.sub(lambda m: " " * len(m.group()), masked)
        tokens = list(WORD.finditer(masked))
        marker_iter = iter(sorted(markers))
        next_marker = next(marker_iter, None)
        for match in tokens:
            while next_marker is not None and next_marker[0] < match.start():
                pending.append(next_marker[2])
                if stamps.get(next_marker[0]):
                    stamp_pending.append({**stamps[next_marker[0]], "side": "before"})
                next_marker = next(marker_iter, None)
            raw = match.group()
            if PARTICIPANT.match(raw):
                # "P7." stands in for a participant's name: the audio holds
                # speech the text does not, so flag it instead of aligning it.
                pending.append("redacted")
                continue
            if not any(ch.isalnum() for ch in raw):
                # Punctuation-only token: keep it with the neighbouring word.
                if turn_words:
                    turn_words[-1].raw += raw
                else:
                    prefix += raw
                continue
            raw, prefix = prefix + raw, ""
            word = Word(
                index=len(words) + len(turn_words), turn_index=turn_index,
                speaker=turn.get("speaker_id"), raw=raw,
                norm=normalize_word(raw), align=align_form(raw, alphabet),
                lang=_word_lang(text, match.start(), match.end(), spans),
                annotations=pending, markers=stamp_pending,
            )
            pending = []
            stamp_pending = []
            turn_words.append(word)
        trailing: List[Dict] = []
        while next_marker is not None:
            pending.append(next_marker[2])
            if stamps.get(next_marker[0]):
                trailing.append({**stamps[next_marker[0]], "side": "after"})
            next_marker = next(marker_iter, None)
        if turn_words:
            turn_words[-1].annotations.extend(pending)
            turn_words[-1].markers.extend(trailing)
        else:
            # A turn that is only a marker: check it against the previous word
            # (its end should come before the marker) and the next one.
            if words:
                words[-1].markers.extend({**m, "side": "after"} for m in stamp_pending + trailing)
            carried = [{**m, "side": "before"} for m in stamp_pending + trailing]
        words.extend(turn_words)
    return words
