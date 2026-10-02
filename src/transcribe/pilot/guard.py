"""Detect and repair Whisper's failure modes on hard audio: repetition loops ("kwa kwa kwa ..."),
letters from unrelated scripts, and low-confidence output. Pure text functions, no model needed."""

import re
import unicodedata
from typing import List, Optional, Tuple

MIN_REPEATS = 4                # a word or phrase repeated this many times in a row is a loop
MAX_PHRASE = 8                 # longest repeating phrase looked for, in words
LOW_LOGPROB = -1.0             # Whisper's own threshold for "probably wrong"


def _norm(word: str) -> str:
    return re.sub(r"[^\w]+", "", word.lower())


def _runs(words: List[str]) -> List[Tuple[int, int, int]]:
    """(start, phrase length, repeats) of each loop found, left to right, non-overlapping."""
    keys = [_norm(w) for w in words]
    found, i = [], 0
    while i < len(keys):
        best = None
        for n in range(1, MAX_PHRASE + 1):
            if i + n * MIN_REPEATS > len(keys):
                break
            phrase = keys[i:i + n]
            if not any(phrase):
                continue
            repeats = 1
            while keys[i + repeats * n:i + (repeats + 1) * n] == phrase:
                repeats += 1
            if repeats >= MIN_REPEATS and (best is None or repeats * n > best[1] * best[2]):
                best = (i, n, repeats)
        if best:
            found.append(best)
            i += best[1] * best[2]
        else:
            i += 1
    return found


def has_loop(text: str) -> bool:
    return bool(_runs(text.split()))


def collapse_loops(text: str) -> str:
    """Keep one copy of every looped word or phrase."""
    words = text.split()
    out, i = [], 0
    for start, n, repeats in _runs(words):
        out += words[i:start + n]
        i = start + n * repeats
    return " ".join(out + words[i:])


def strip_stray_scripts(text: str) -> str:
    """Remove letters that are not Latin (Korean, Chinese, Arabic ... leak into noisy chunks);
    punctuation, digits and accented Latin letters stay."""
    def keep(ch: str) -> bool:
        if not ch.isalpha():
            return unicodedata.category(ch) != "So" and ch != "�"
        return unicodedata.name(ch, "").startswith("LATIN")
    return re.sub(r"\s{2,}", " ", "".join(ch for ch in text if keep(ch))).strip()


def repair(text: str, avg_logprob: Optional[float] = None) -> Tuple[str, List[str]]:
    """Cleaned text and the reasons (if any) a person should look at this chunk."""
    flags = []
    cleaned = strip_stray_scripts(text)
    if cleaned != text.strip():
        flags.append("stray-script")
    if has_loop(cleaned):
        cleaned = collapse_loops(cleaned)
        flags.append("loop")
    if avg_logprob is not None and avg_logprob < LOW_LOGPROB:
        flags.append("low-confidence")
    return cleaned, flags
