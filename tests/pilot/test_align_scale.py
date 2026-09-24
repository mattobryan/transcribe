"""Regression: a long interview with unspoken turns and untranscribed chatter."""
import numpy as np

from src.transcribe.pilot.ctc_align import align_words
from src.transcribe.pilot.textprep import turns_to_words

LETTERS = "abcdefghijklmnopqrstuvwxyz'"
VOCAB = {c: i + 1 for i, c in enumerate(LETTERS)}
SYLLABLES = ["ha", "ba", "ri", "ya", "ko", "ni", "me", "sa", "na", "wa", "tu", "li", "e", "nda",
             "ka", "si", "the", "and", "you", "form", "time", "job"]


def test_recovers_after_unspoken_turns():
    rng = np.random.default_rng(7)
    turns, labels, spoken = [], [0] * 50, []
    for n in range(200):
        count = int(rng.integers(2, 12)) if rng.random() > 0.05 else int(rng.integers(60, 120))
        text = " ".join("".join(rng.choice(SYLLABLES, size=rng.integers(1, 4))) for _ in range(count))
        turns.append({"text": text, "speaker_id": "I" if n % 2 == 0 else "R", "spans": []})
        if rng.random() < 0.06:                    # summarised by the transcriber
            spoken.append(False)
            continue
        if rng.random() < 0.04:                    # speech missing from the transcript
            labels += [VOCAB[c] for c in "zqxzqx" * 20 for _ in range(3)] + [0] * 30
        for word in text.split():
            for ch in word:
                labels += [VOCAB[ch]] * 2 + [0]
            labels += [0] * int(rng.integers(3, 12))
        labels += [0] * int(rng.integers(20, 80))
        spoken.append(True)
    probs = rng.uniform(0, 0.03, size=(len(labels), len(VOCAB) + 1))
    probs[np.arange(len(labels)), labels] = 0.6
    probs /= probs.sum(axis=1, keepdims=True)
    words = turns_to_words(turns, alphabet=LETTERS)
    reports = align_words(np.log(probs), words, lambda s: [VOCAB[c] for c in s if c in VOCAB], 0)
    placed = [r["status"] in ("aligned", "short") for r in reports]
    # every spoken turn is placed and every unspoken anchor turn is rejected
    assert all(p for p, s in zip(placed, spoken) if s)
    assert not any(r["status"] == "aligned" for r, s in zip(reports, spoken) if not s)
