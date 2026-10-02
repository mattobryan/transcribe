import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("el", Path(__file__).parents[2] / "scripts" / "elevenlabs_transcribe.py")
el = importlib.util.module_from_spec(spec)
spec.loader.exec_module(el)


def w(text, start, end, speaker="speaker_0", type="word"):
    return {"text": text, "start": start, "end": end, "type": type, "speaker_id": speaker}


def test_turns_split_on_speaker_and_pause():
    reply = {"words": [w("Habari", 0, 0.5), w(" ", 0.5, 0.6, type="spacing"), w("yako", 0.6, 1.0),
                       w("Nzuri", 1.2, 1.6, "speaker_1"), w("sana", 5.0, 5.4, "speaker_1")]}
    items = el.turns(reply)
    assert [c["text"] for c in items] == ["Habari yako", "Nzuri", "sana"]
    txt, srt = el.render(reply)
    assert txt.startswith("[0:00:00] speaker_0: Habari yako") and "00:00:01,200 --> 00:00:01,600" in srt
