import csv
import json

import numpy as np
import soundfile as sf

from transcribe.train import dataset
from transcribe.train.evaluate import report_markdown, scribe_hypotheses, score_system


def test_training_text_drops_labels_marks_and_dashes_but_keeps_hyphenated_words():
    final = "I: Or-\nR: I work at **Mbagathi Hospital** in _ni_break_ie_ now... one-on-one, uko-- You're down."
    assert dataset.training_text(final) == "Or I work at Mbagathi Hospital in nibreakie now one-on-one, uko You're down."


def test_reference_and_dominant_language():
    rows = [{"token": "Habari", "language": "sw"}, {"token": "yako,", "language": "sw"}, {"token": "friend.", "language": "en"},
            {"token": "nimeapply", "language": "mixed"}, {"token": "Nyale", "language": "name"}]
    words, langs, speakers = dataset.reference(rows)
    assert words == ["habari", "yako", "friend", "nimeapply", "nyale"] and langs == ["sw", "sw", "en", "mixed", "name"]
    assert speakers == ["?"] * 5
    assert dataset.dominant_language(langs) == "sw" and dataset.dominant_language(["en", "sw"]) == "en"


def make_recording(tmp_path):
    wav = tmp_path / "talk.wav"
    sf.write(wav, np.zeros(16000 * 40, dtype="float32"), 16000)
    chunks = [
        {"chunk": 1, "start": 0.0, "end": 8.0, "checked": True, "changed": True, "scribe_raw": [{"speaker": "Speaker 1", "text": "Habari yako friend"}],
         "after_automatic_clean_up": "I: Habari yako friend", "final": "I: Habari yako rafiki."},
        {"chunk": 2, "start": 8.0, "end": 15.0, "checked": False, "changed": False, "scribe_raw": [], "after_automatic_clean_up": "R: Nzuri", "final": "R: Nzuri"},
        {"chunk": 3, "start": 15.0, "end": 16.0, "checked": True, "changed": False, "scribe_raw": [{"speaker": "Speaker 2", "text": "Mm-hmm"}],
         "after_automatic_clean_up": "R: Mm-hmm.", "final": "R: Mm-hmm."},
    ]
    log = tmp_path / "log.json"
    log.write_text(json.dumps({"chunks": chunks}))
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["recording", "chunk", "start", "end", "speaker", "token", "language", "parts", "source"])
        for tok, lang in [("Habari", "sw"), ("yako", "sw"), ("rafiki.", "sw")]:
            w.writerow(["R1", 1, 0, 8, "I", tok, lang, "", "automatic"])
        w.writerow(["R1", 3, 15, 16, "R", "Mm-hmm.", "en", "", "automatic"])
    return wav, {"name": "Interview 1", "log": str(log), "labels": str(labels), "split": "dev"}


def test_build_rows_uses_only_checked_chunks_and_cuts_clips(tmp_path):
    wav, rec = make_recording(tmp_path)
    rows = dataset.build_rows(rec, wav, tmp_path / "out" / "clips", base=tmp_path / "out")
    assert [r["chunk"] for r in rows] == [1, 3]                    # chunk 2 was never checked
    first = rows[0]
    assert first["text"] == "Habari yako rafiki." and first["ref_words"] == ["habari", "yako", "rafiki"] and first["language"] == "sw"
    assert first["audio"] == "clips/clip_00001.wav".replace("clip_", "interview_1_")
    counts = dataset.write_manifests(rows, tmp_path / "out")
    first = dataset.read_manifest(tmp_path / "out" / "dev.jsonl")[0]          # read back: the path now points at the real file
    data, rate = sf.read(first["audio"])
    assert rate == 16000 and abs(len(data) / rate - 8.0) < 0.01
    assert first["scribe_raw"] == "Habari yako friend" and first["scribe_cleaned"] == "Habari yako friend"
    assert counts == {"train": 0, "dev": 2, "test": 0}
    assert len(dataset.read_manifest(tmp_path / "out" / "dev.jsonl")) == 2


def test_scribe_baseline_is_scored_by_language(tmp_path):
    wav, rec = make_recording(tmp_path)
    rows = dataset.build_rows(rec, wav, tmp_path / "clips")
    raw = score_system(rows, scribe_hypotheses(rows, "scribe_raw"))
    # chunk 1: "friend" instead of "rafiki" is one Swahili word wrong out of 3; chunk 3 "Mm-hmm" is right
    assert raw["ref_words"] == 4 and raw["wer"] == round(1 / 4, 4)
    assert raw["wer_by_lang"]["sw"] == round(1 / 3, 4)
    assert raw["wer_by_speaker"] == {"I": round(1 / 3, 4), "R": 0.0}
    assert "| scribe_raw |" in report_markdown({"scribe_raw": raw}, "dev", 2)
