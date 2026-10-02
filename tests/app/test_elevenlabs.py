import json

from transcribe.app import elevenlabs
from transcribe.app.store import Store


def w(text, start, end, speaker="speaker_0", logprob=0.0, type="word"):
    return {"text": text, "start": start, "end": end, "type": type, "speaker_id": speaker, "logprob": logprob}


def reply(n=40):
    words, t = [], 0.0
    for i in range(n):
        speaker = "speaker_0" if i < n // 2 else "speaker_1"
        words += [w(f"w{i}", t, t + 0.5, speaker, -2.0 if i == 3 else 0.0), w(" ", t + 0.5, t + 0.8, speaker, type="spacing")]
        t += 0.8
    return {"words": words}


def test_chunks_split_on_speaker_and_label_in_order():
    items = elevenlabs.chunks(reply())
    assert [c["speaker"] for c in items] == ["Speaker 1", "Speaker 2"]
    assert items[0]["text"].startswith("w0 w1") and items[0]["unsure"] > 0 and items[1]["unsure"] == 0


def test_long_run_is_cut_at_max_length():
    items = elevenlabs.chunks({"words": reply(200)["words"][:200]})
    assert all(c["end"] - c["start"] <= elevenlabs.MAX_CHUNK_S + 1e-6 for c in items)


def test_import_creates_session_with_audio(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    audio = tmp_path / "talk.wav"
    sf.write(audio, np.zeros(16000 * 30, dtype="float32"), 16000)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(reply()))
    store = Store(tmp_path / "s")
    sid = elevenlabs.import_elevenlabs(store, path, audio)
    session = store.load(sid)
    assert session["status"] == "ready" and len(session["segments"]) == 2
    assert session["segments"][0]["speaker_id"] == "Speaker 1" and (store.folder(sid) / "audio16k.wav").exists()
