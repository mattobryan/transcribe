from src.transcribe.pilot.segment import Policy, build_segments, group_words, silence_gaps
from src.transcribe.pilot.textprep import Word


def _w(i, start, end, turn=0, speaker="respondent", score=0.9, raw=None, lang="sw", ann=None):
    raw = raw or f"w{i}"
    return Word(index=i, turn_index=turn, speaker=speaker, raw=raw, norm=raw.lower(), align=raw,
                lang=lang, annotations=ann or [], start=start, end=end, score=score)


def test_never_exceeds_max_and_keeps_every_word():
    words = [_w(i, i * 0.5, i * 0.5 + 0.4) for i in range(200)]     # 100 s, 0.1 s gaps
    groups = group_words(words, Policy())
    assert sum(len(g) for g in groups) == 200
    assert all(g[-1].end - g[0].start <= 30.0 for g in groups)


def test_splits_at_pause_after_target_length():
    words = [_w(i, i * 0.5, i * 0.5 + 0.45) for i in range(20)]        # 10 s continuous
    words += [_w(20 + i, 10.6 + i * 0.5, 10.6 + i * 0.5 + 0.45) for i in range(10)]
    groups = group_words(words, Policy())
    assert len(groups[0]) == 20


def test_speaker_change_starts_new_segment():
    words = [_w(i, i * 0.5, i * 0.5 + 0.45, turn=0, speaker="interviewer") for i in range(6)]
    words += [_w(6 + i, 3.1 + i * 0.5, 3.1 + i * 0.5 + 0.45, turn=1) for i in range(6)]
    groups = group_words(words, Policy())
    assert [g[0].speaker for g in groups] == ["interviewer", "respondent"]


def test_flags_and_fields():
    words = [_w(0, 1.0, 1.4, raw="Sawa", score=0.3), _w(1, 1.5, 2.0, raw="2019", ann=["unclear"]),
             _w(2, 2.1, 2.8, raw="sasa", score=0.2)]
    seg = build_segments(words, "rec", 60.0)[0]
    assert seg["text"] == "Sawa 2019 sasa"
    assert seg["ref_words"] == ["sawa", "2019", "sasa"]
    assert {"low_alignment", "unclear_marker", "has_number"} <= set(seg["flags"])
    assert seg["start"] == 0.9 and seg["end"] == 2.9


def test_silence_gaps():
    words = [_w(0, 0, 1), _w(1, 5, 6), _w(2, 6.1, 7, score=0.2), _w(3, 12, 13)]
    gaps = silence_gaps(words)
    assert [(g["start"], g["end"]) for g in gaps] == [(1.2, 4.8)]
