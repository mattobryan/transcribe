from transcribe.app.adapt import apply_rules, learn, word_pairs


def test_word_pairs():
    pairs, kept = word_pairs("ni kwa sababa ya maji", "Ni kwa sababu ya maji.")
    assert pairs == [("sababa", "sababu")]
    assert kept == ["ni", "kwa", "ya", "maji"]


def segs():
    return [
        {"asr_hypothesis": "kwa sababa ya mvua", "transcript": "kwa sababu ya mvua", "edited": True},
        {"asr_hypothesis": "sababa ni Ochieng", "transcript": "sababu ni Achieng'", "edited": True},
        {"asr_hypothesis": "hiyo sababa", "transcript": "hiyo sababa", "edited": False},
    ]


def test_rule_needs_two_consistent_corrections():
    learned = learn(segs()[:1])
    assert learned["rules"] == []
    learned = learn(segs())
    assert learned["rules"] == [["sababa", "sababu", 2]]


def test_rule_not_learned_when_word_often_kept():
    many_kept = segs() + [{"asr_hypothesis": "sababa", "transcript": "sababa", "edited": True}] * 3
    assert learn(many_kept)["rules"] == []


def test_prompt_and_hotwords():
    learned = learn(segs())
    assert learned["prompt"].endswith("sababu ni Achieng'")
    assert "Achieng'" in learned["hotwords"]
    assert "sababu" in learned["hotwords"]            # typed, never produced by the model
    assert "mvua" not in learned["hotwords"]          # the model already knew it


def test_apply_rules_keeps_case_and_punctuation():
    rules = [["sababa", "sababu", 2]]
    assert apply_rules("Sababa, hiyo ni sababa.", rules) == "Sababu, hiyo ni sababu."
    assert apply_rules("hakuna kitu", rules) == "hakuna kitu"
