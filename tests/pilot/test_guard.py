import numpy as np

from transcribe.pilot import asr, guard


def test_detects_and_collapses_word_loop():
    text = "So we go. kwa kwa kwa kwa kwa kwa kwa kwa and then"
    assert guard.has_loop(text)
    assert guard.collapse_loops(text) == "So we go. kwa and then"


def test_phrase_loop_and_number_loop():
    assert guard.collapse_loops("start nasiha kwa 2 nasiha kwa 2 nasiha kwa 2 nasiha kwa 2 nasiha kwa 2") == \
        "start nasiha kwa 2"
    assert guard.collapse_loops("3, 3, 5, 5, 5, 5, 5, 5") == "3, 3, 5,"


def test_normal_speech_is_untouched():
    text = "ya ya, I I think we go one by one, haja haja haja."
    assert not guard.has_loop(text)
    assert guard.repair(text) == (text, [])


def test_stray_scripts_removed_and_flagged():
    text, flags = guard.repair("ukinia kwa kulim려 ya wikiwa, wa أعرض not 變utiku �")
    assert text == "ukinia kwa kulim ya wikiwa, wa not utiku"
    assert flags == ["stray-script"]


def test_low_confidence_flag_and_accents_kept():
    text, flags = guard.repair("café ni nzuri", avg_logprob=-1.4)
    assert text == "café ni nzuri" and flags == ["low-confidence"]


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        looping = "repetition_penalty" not in kw
        text = "kwa " * 30 if looping else "habari ya asubuhi"
        seg = type("S", (), {"text": text, "avg_logprob": -0.3, "no_speech_prob": 0.0})()
        info = type("I", (), {"language": "sw", "language_probability": 0.9})()
        return [seg], info


def runner_with(model):
    r = asr.WhisperRunner.__new__(asr.WhisperRunner)
    r.device, r.model = "cpu", model
    return r


def test_loop_triggers_retry_with_repetition_penalty():
    model = FakeModel()
    out = runner_with(model).transcribe(np.zeros(16000, np.float32), "sw")
    assert out["text"] == "habari ya asubuhi"
    assert "flag" not in out
    assert len(model.calls) == 2 and model.calls[1]["repetition_penalty"] > 1


def test_loop_that_survives_retry_is_collapsed_and_flagged():
    class Stubborn(FakeModel):
        def transcribe(self, audio, **kw):
            seg = type("S", (), {"text": "kwa " * 30, "avg_logprob": -0.3, "no_speech_prob": 0.0})()
            info = type("I", (), {"language": "sw", "language_probability": 0.9})()
            return [seg], info
    out = runner_with(Stubborn()).transcribe(np.zeros(16000, np.float32), "sw")
    assert out["text"] == "kwa" and out["flag"] == ["loop"]
