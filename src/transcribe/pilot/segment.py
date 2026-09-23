"""Group aligned words into training-sized segments (TRD section 3)."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .textprep import Word

MAX_S = 30.0
MIN_S = 1.5
TARGET_S = 8.0
SPLIT_GAP_S = 0.3
LONG_PAUSE_S = 1.0
HARD_GAP_S = 2.0
PAD_S = 0.1
LOW_ALIGNMENT = 0.5


@dataclass
class Policy:
    max_s: float = MAX_S
    min_s: float = MIN_S
    target_s: float = TARGET_S
    split_gap_s: float = SPLIT_GAP_S
    long_pause_s: float = LONG_PAUSE_S
    hard_gap_s: float = HARD_GAP_S
    low_alignment: float = LOW_ALIGNMENT


def _duration(words: List[Word]) -> float:
    return words[-1].end - words[0].start


def _best_split(words: List[Word], policy: Policy) -> int:
    """Index where the largest pause leaves at least min_s on the left."""
    best, best_gap = len(words), -1.0
    for i in range(1, len(words)):
        if words[i - 1].end - words[0].start < policy.min_s:
            continue
        gap = words[i].start - words[i - 1].end
        if gap > best_gap:
            best, best_gap = i, gap
    return best


def group_words(words: List[Word], policy: Policy = Policy()) -> List[List[Word]]:
    timed = [w for w in words if w.start is not None and w.end is not None]
    groups: List[List[Word]] = []
    current: List[Word] = []
    for word in timed:
        if current:
            gap = word.start - current[-1].end
            length = _duration(current)
            if gap >= policy.hard_gap_s or word.turn_index < current[-1].turn_index:
                groups.append(current)
                current = []
            elif word.speaker != current[-1].speaker and length >= policy.min_s:
                groups.append(current)
                current = []
            elif word.end - current[0].start > policy.max_s:
                cut = _best_split(current, policy)
                groups.append(current[:cut])
                current = current[cut:]
            elif gap >= policy.split_gap_s and length >= policy.target_s:
                groups.append(current)
                current = []
            elif gap >= policy.long_pause_s and length >= policy.min_s:
                groups.append(current)
                current = []
        current.append(word)
        # A single cut may still leave an over-long remainder.
        while current and _duration(current) > policy.max_s and len(current) > 1:
            cut = _best_split(current, policy)
            if cut >= len(current):
                cut = len(current) - 1
            groups.append(current[:cut])
            current = current[cut:]
    if current:
        groups.append(current)
    return groups


def build_segments(words: List[Word], recording_id: str, audio_seconds: float,
                   policy: Policy = Policy()) -> List[Dict]:
    groups = group_words(words, policy)
    segments = []
    for n, group in enumerate(groups):
        prev_end = groups[n - 1][-1].end if n else 0.0
        next_start = groups[n + 1][0].start if n + 1 < len(groups) else audio_seconds
        start = max(group[0].start - PAD_S, (prev_end + group[0].start) / 2, 0.0)
        end = min(group[-1].end + PAD_S, (group[-1].end + next_start) / 2, audio_seconds)
        scores = [w.score for w in group if w.score is not None]
        confidence = float(np.mean(scores)) if scores else 0.0
        flags = []
        if confidence < policy.low_alignment:
            flags.append("low_alignment")
        kinds = {kind for w in group for kind in w.annotations}
        flags.extend(f"{kind}_marker" for kind in sorted(kinds))
        if any(w.has_digit for w in group):
            flags.append("has_number")
        if len({w.speaker for w in group}) > 1:
            flags.append("mixed_speaker")
        if end - start < policy.min_s:
            flags.append("too_short")
        if end - start > policy.max_s + 2 * PAD_S:
            flags.append("too_long")
        speakers = [w.speaker for w in group]
        segments.append({
            "segment_id": f"{recording_id}_{n:05d}",
            "sequence": n,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "text": " ".join(w.raw for w in group),
            "ref_words": [w.norm for w in group if w.norm],
            "ref_langs": [w.lang for w in group if w.norm],
            "speaker": max(set(speakers), key=speakers.count),
            "turn_indices": sorted({w.turn_index for w in group}),
            "alignment_conf": round(confidence, 4),
            "min_word_conf": round(min(scores), 4) if scores else 0.0,
            "flags": flags,
        })
    return segments


def silence_gaps(words: List[Word], min_gap: float = 2.5, min_conf: float = 0.5,
                 limit: int = 40) -> List[Dict]:
    """Untranscribed stretches between confidently aligned words.

    Used to measure hallucination: any words a model emits here are either
    hallucinated or speech missing from the transcript.
    """
    timed = [w for w in words if w.start is not None and w.score is not None]
    gaps = []
    for a, b in zip(timed, timed[1:]):
        if a.score >= min_conf and b.score >= min_conf and b.start - a.end >= min_gap:
            start, end = a.end + 0.2, min(b.start - 0.2, a.end + 0.2 + 30.0)
            gaps.append({"start": round(start, 3), "end": round(end, 3),
                         "after_turn": a.turn_index})
    gaps.sort(key=lambda g: g["end"] - g["start"], reverse=True)
    return sorted(gaps[:limit], key=lambda g: g["start"])
