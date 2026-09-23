import numpy as np

from src.transcribe.pilot.ctc_align import viterbi_align

BLANK = 0


def _logp(frames, vocab, labels):
    """labels: frame -> dominant token (default blank)."""
    probs = np.full((frames, vocab), 0.02)
    for t in range(frames):
        probs[t, labels.get(t, BLANK)] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


def test_finds_tokens_inside_other_speech():
    labels = {t: 4 for t in range(0, 30)}          # neighbouring turn's speech
    labels.update({t: 1 for t in range(40, 43)})
    labels.update({t: 2 for t in range(45, 48)})
    labels.update({t: 3 for t in range(50, 53)})
    labels.update({t: 4 for t in range(60, 100)})
    spans = viterbi_align(_logp(100, 5, labels), [1, 2, 3], BLANK)
    assert [(s.start, s.end) for s in spans] == [(40, 43), (45, 48), (50, 53)]
    assert all(np.exp(s.logprob) > 0.8 for s in spans)


def test_repeated_token_needs_blank():
    labels = {10: 1, 11: 1, 12: 0, 13: 1}
    spans = viterbi_align(_logp(20, 3, labels), [1, 1], BLANK)
    assert spans[0].end <= spans[1].start
    assert spans[1].start == 13


def test_too_short_window():
    assert viterbi_align(_logp(2, 3, {}), [1, 2, 1], BLANK) is None
    assert viterbi_align(_logp(5, 3, {}), [], BLANK) is None
