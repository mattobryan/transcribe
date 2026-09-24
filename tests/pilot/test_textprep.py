from dataclasses import asdict

from src.transcribe.pilot.textprep import (
    align_form, normalize_text, normalize_word, select_turns, turns_to_words)
from src.transcribe.preprocessing.docx_parser import parse_docx


def _words(path):
    turns = select_turns([asdict(t) for t in parse_docx(str(path)).transcript_turns])
    return turns, turns_to_words(turns)


def test_header_dropped_and_continuation_kept(interview_docx):
    turns, _ = _words(interview_docx)
    assert [t["speaker_id"] for t in turns] == ["interviewer", "respondent", "respondent"]
    assert turns[2]["text"] == "and then I left"


def test_language_tags_from_italics(interview_docx):
    _, words = _words(interview_docx)
    tags = {w.norm: w.lang for w in words}
    assert tags["habari"] == "sw"
    assert tags["yako"] == "sw"
    assert tags["how"] == "en"
    assert tags["nimedownload"] == "mixed"
    assert tags["fine"] == "en"


def test_annotation_attached_to_next_word_and_removed(interview_docx):
    _, words = _words(interview_docx)
    raws = [w.raw for w in words]
    assert "[unclear]" not in " ".join(raws)
    marked = [w.raw for w in words if w.annotations]
    assert marked == ["nimedownload"]
    assert words[raws.index("nimedownload")].annotations == ["unclear"]


def test_digits_are_not_alignable_but_scored():
    assert align_form("2019.") == ""
    assert normalize_word("2019.") == "2019"
    assert align_form("Nime-download,") == "nimedownload"


def test_normalizer():
    assert normalize_word("Nime-download,") == "nimedownload"
    assert normalize_word("don’t") == "don't"
    assert normalize_text("Sawa [laughs] OK, fine!") == ["sawa", "ok", "fine"]


def test_alphabet_filter():
    assert align_form("café", alphabet="acf") == "caf"


def test_punctuation_only_tokens_join_neighbours():
    words = turns_to_words([{"text": "“ Habari , yako . ”", "speaker_id": "R", "spans": []}])
    assert [w.raw for w in words] == ["“Habari,", "yako.”"]


def test_false_starts_split():
    words = turns_to_words([{"text": "Haya tuta--tutaanza, tuna--, kunasa\u2026 sawa", "speaker_id": "I", "spans": []}])
    assert [w.norm for w in words] == ["haya", "tuta", "tutaanza", "tuna", "kunasa", "sawa"]


def test_real_transcript_markers():
    text = "Na watu wa (Unclear: 1:38:47-1:38:49]]. Sawa [No speech - activity, 17:22-17:24] haya [[Unclear 1:16:07]]"
    words = turns_to_words([{"text": text, "speaker_id": "R", "spans": []}])
    assert [w.norm for w in words] == ["na", "watu", "wa", "sawa", "haya"]
    assert words[3].annotations == ["unclear"]
    assert words[4].annotations == ["no_speech", "unclear"]
