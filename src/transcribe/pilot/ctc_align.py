"""Turn-anchored CTC forced alignment of a long recording to its transcript.

Emissions come from a multilingual CTC aligner (MMS-300m by default). Each
transcript turn is aligned inside a search window that follows the previous
turn, so the Viterbi trellis stays small and a non-verbatim passage only
damages its own turn. Garbage states before and after the turn absorb audio
that belongs to neighbouring turns.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .textprep import Word

FRAME_SECONDS = 0.02            # wav2vec2-style models emit one frame per 20 ms
SAMPLES_PER_FRAME = 320         # at 16 kHz
GARBAGE_PENALTY = 0.01          # nats; only breaks ties in favour of the turn
DEFAULT_ALIGNER = "MahmoudAshraf/mms-300m-1130-forced-aligner"


@dataclass
class TokenSpan:
    start: int                  # first frame (inclusive)
    end: int                    # last frame (exclusive)
    logprob: float              # mean log-probability over the token's frames


def viterbi_align(logp: np.ndarray, tokens: Sequence[int], blank: int) -> Optional[List[TokenSpan]]:
    """Align ``tokens`` somewhere inside ``logp`` (frames x vocab log-probs).

    Leading and trailing frames may be absorbed by garbage states whose score
    is the best log-probability in that frame. Returns one span per token, or
    None if the window is too short to hold the tokens.
    """
    frames = logp.shape[0]
    n_tokens = len(tokens)
    if n_tokens == 0 or frames < n_tokens:
        return None
    ext = np.empty(2 * n_tokens + 1, dtype=np.int64)
    ext[0::2] = blank
    ext[1::2] = tokens
    n_ext = len(ext)
    neg = -1e30
    # Garbage is slightly worse than the best token so real speech frames are
    # kept by the turn's own tokens instead of being absorbed at the edges.
    garbage = logp.max(axis=1) - GARBAGE_PENALTY
    skip_ok = np.zeros(n_ext, dtype=bool)
    skip_ok[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    # States: 0 = leading garbage, 1..n_ext = extended CTC states, n_ext+1 = trailing garbage.
    last = n_ext + 1
    score = np.full(n_ext + 2, neg)
    score[0] = garbage[0]
    first = logp[0, ext]
    score[1] = first[0]
    score[2] = first[1]
    back = np.zeros((frames, n_ext + 2), dtype=np.int8)
    for t in range(1, frames):
        prev = score
        new = np.full(n_ext + 2, neg)
        new[0] = prev[0] + garbage[t]
        stay = prev[1:n_ext + 1]
        step1 = prev[0:n_ext]
        step2 = np.full(n_ext, neg)
        step2[2:] = np.where(skip_ok[2:], prev[1:n_ext - 1], neg)
        step2[1] = prev[0]      # leading garbage straight into the first token
        options = np.stack([stay, step1, step2])
        choice = options.argmax(axis=0)
        new[1:n_ext + 1] = options[choice, np.arange(n_ext)] + logp[t, ext]
        back[t, 1:n_ext + 1] = choice
        tail = (prev[last], prev[n_ext], prev[n_ext - 1])
        best_tail = int(np.argmax(tail))
        new[last] = tail[best_tail] + garbage[t]
        back[t, last] = best_tail
        score = new

    finals = (score[last], score[n_ext], score[n_ext - 1])
    state = (last, n_ext, n_ext - 1)[int(np.argmax(finals))]
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        move = int(back[t, state])
        if state == last:
            state = (last, n_ext, n_ext - 1)[move]
        else:
            state = state - move

    spans: List[Optional[List[float]]] = [None] * n_tokens
    for t, state in enumerate(path):
        if 1 <= state <= n_ext and (state - 1) % 2 == 1:
            k = (state - 2) // 2
            value = logp[t, tokens[k]]
            if spans[k] is None:
                spans[k] = [t, t + 1, value, 1]
            else:
                spans[k][1] = t + 1
                spans[k][2] += value
                spans[k][3] += 1
    if any(span is None for span in spans):
        return None
    return [TokenSpan(int(s[0]), int(s[1]), float(s[2] / s[3])) for s in spans]


class Emitter:
    """Computes frame log-probabilities with a Hugging Face CTC model."""

    def __init__(self, model_name: str = DEFAULT_ALIGNER, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCTC, AutoTokenizer

        self.torch = torch
        self.device = device
        self.model = AutoModelForCTC.from_pretrained(model_name).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab: Dict[str, int] = tokenizer.get_vocab()
        blank = self.vocab.get("<blank>")
        if blank is None:
            blank = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.blank = int(blank)
        letters = [t for t in self.vocab if len(t) == 1 and t.isalpha()]
        if len(letters) < 20:
            raise RuntimeError(
                f"Aligner {model_name} has an unexpected vocabulary "
                f"({sorted(self.vocab)[:40]}); expected single romanised letters.")
        self.do_normalize = True
        try:
            from transformers import AutoFeatureExtractor
            self.do_normalize = bool(getattr(
                AutoFeatureExtractor.from_pretrained(model_name), "do_normalize", True))
        except Exception:
            pass

    @property
    def alphabet(self) -> List[str]:
        return [token for token in self.vocab if len(token) == 1 and (token.isalpha() or token == "'")]

    def token_ids(self, text: str) -> List[int]:
        return [self.vocab[ch] for ch in text if ch in self.vocab]

    def emissions(self, audio: np.ndarray, window_s: float = 30.0, context_s: float = 2.0,
                  progress=None) -> np.ndarray:
        """Log-probabilities (frames x vocab) for the whole recording, windowed."""
        torch = self.torch
        sr = 16000
        total_frames = int(math.ceil(len(audio) / SAMPLES_PER_FRAME))
        out = None
        step = int(window_s * sr)
        ctx = int(context_s * sr)
        starts = list(range(0, len(audio), step))
        for i, start in enumerate(starts):
            lo = max(0, start - ctx)
            hi = min(len(audio), start + step + ctx)
            chunk = audio[lo:hi].astype(np.float32)
            if self.do_normalize:
                chunk = (chunk - chunk.mean()) / np.sqrt(chunk.var() + 1e-7)
            with torch.inference_mode():
                logits = self.model(torch.from_numpy(chunk)[None].to(self.device)).logits[0]
                logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            if out is None:
                out = np.full((total_frames, logp.shape[1]), -1e4, dtype=np.float32)
                out[:, self.blank] = 0.0
            first = start // SAMPLES_PER_FRAME
            last = min(total_frames, (start + step) // SAMPLES_PER_FRAME)
            offset = lo // SAMPLES_PER_FRAME
            for frame in range(first, last):
                local = frame - offset
                if 0 <= local < len(logp):
                    out[frame] = logp[local]
            if progress:
                progress(i + 1, len(starts))
        return out


def load_or_compute_emissions(emitter: Emitter, audio: np.ndarray, cache: Path, progress=None) -> np.ndarray:
    if cache.exists():
        return np.load(cache).astype(np.float32)
    logp = emitter.emissions(audio, progress=progress)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, logp.astype(np.float16))
    return logp


def align_words(logp: np.ndarray, words: List[Word], token_ids, blank: int,
                chars_per_second: float = 14.0, trust: float = 0.5,
                max_window_s: float = 900.0, progress=None) -> List[Dict]:
    """Align words turn by turn; fills word.start/end/score in place.

    The search cursor only moves forward after a trusted turn (confidence >=
    ``trust``). A low-confidence turn is usually text that was never spoken
    (summarised or skipped by the transcriber), so the next turn searches from
    the same cursor with a window widened by the skipped turns' expected length.

    Returns one record per turn with its window, confidence and status.
    """
    frames_per_second = 1.0 / FRAME_SECONDS
    total = logp.shape[0]
    by_turn: Dict[int, List[Word]] = {}
    for word in words:
        by_turn.setdefault(word.turn_index, []).append(word)
    cursor = 0
    pending = 0.0               # expected frames of untrusted turns since the cursor
    reports = []
    turn_ids = sorted(by_turn)
    for n, turn_index in enumerate(turn_ids):
        turn_words = by_turn[turn_index]
        alignable = [w for w in turn_words if w.align]
        ids: List[int] = []
        owners: List[int] = []
        for position, word in enumerate(alignable):
            word_ids = token_ids(word.align)
            ids.extend(word_ids)
            owners.extend([position] * len(word_ids))
        record = {"turn_index": turn_index, "speaker": turn_words[0].speaker,
                  "n_words": len(turn_words), "status": "empty", "confidence": None}
        if ids:
            expected = len(ids) / chars_per_second * frames_per_second
            lo = max(0, cursor - int(0.5 * frames_per_second))
            reach = max(30 * frames_per_second, 2.5 * expected) + 2.5 * pending + 10 * frames_per_second
            hi = min(total, cursor + int(min(reach, max_window_s * frames_per_second)))
            if hi - lo < len(ids) * 2:
                hi = min(total, lo + len(ids) * 3)
            spans = viterbi_align(logp[lo:hi], ids, blank)
            if spans is None:
                record["status"] = "failed"
                pending += expected
            else:
                per_word: Dict[int, List[TokenSpan]] = {}
                for owner, span in zip(owners, spans):
                    per_word.setdefault(owner, []).append(span)
                probs = []
                for position, word in enumerate(alignable):
                    word_spans = per_word[position]
                    word.start = (lo + word_spans[0].start) * FRAME_SECONDS
                    word.end = (lo + word_spans[-1].end) * FRAME_SECONDS
                    word.score = float(np.mean([math.exp(s.logprob) for s in word_spans]))
                    probs.append(word.score)
                confidence = float(np.mean(probs))
                record.update(status="aligned", confidence=confidence,
                              start=alignable[0].start, end=alignable[-1].end,
                              window=[lo * FRAME_SECONDS, hi * FRAME_SECONDS])
                if confidence >= trust:
                    cursor = lo + spans[-1].end
                    pending = 0.0
                else:
                    # Probably text that was never spoken: keep the turn's
                    # confidence for the report, but give its words no timings
                    # so they cannot become segments on someone else's audio.
                    record["status"] = "suspect"
                    pending += expected
                    for word in alignable:
                        word.start = word.end = word.score = None
        reports.append(record)
        if progress:
            progress(n + 1, len(turn_ids))
    _interpolate(words)
    return reports


def _interpolate(words: List[Word]) -> None:
    """Give unalignable words (digits, symbols) times between their neighbours."""
    for i, word in enumerate(words):
        if word.start is not None:
            continue
        prev_end = next((w.end for w in reversed(words[:i]) if w.end is not None and w.turn_index == word.turn_index), None)
        next_start = next((w.start for w in words[i + 1:] if w.start is not None and w.turn_index == word.turn_index), None)
        if prev_end is None and next_start is None:
            continue
        if prev_end is None:
            prev_end = max(0.0, next_start - 0.3)
        if next_start is None:
            next_start = prev_end + 0.3
        word.start, word.end = prev_end, max(prev_end, next_start)
