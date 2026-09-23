from src.transcribe.pilot.metrics import aggregate, score_segment, switch_points


def test_switch_points():
    assert switch_points(["sw", "sw", "en", "en", "mixed", "sw"]) == [False, True, True, False, True, False]


def test_scores_and_language_attribution():
    ref = ["nilikuwa", "nimeapply", "hiyo", "job", "lakini", "they", "never", "called"]
    langs = ["sw", "mixed", "sw", "en", "sw", "en", "en", "en"]
    seg = {"ref_words": ref, "ref_langs": langs}
    s = score_segment(ref, langs, "Nilikuwa nime apply hiyo job lakini they never called.")
    agg = aggregate([s], [seg])
    # "nimeapply" -> "nime apply": 1 substitution + 1 insertion
    assert s["subs"] == 1 and s["ins"] == 1 and s["dels"] == 0
    assert agg["wer"] == round(2 / 8, 4)
    assert agg["wer_by_lang"]["mixed"] == 1.0
    assert agg["wer_by_lang"]["en"] == 0.0
    assert agg["switch_point_error_rate"] == round(1 / 5, 4)


def test_empty_hypothesis_is_all_deletions():
    s = score_segment(["habari", "yako"], ["sw", "sw"], "")
    assert s["dels"] == 2 and s["errors"] == 2
