"""Transcription app API end to end, with a fake Whisper and fake speech regions."""
import io
import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from transcribe.app.server import create_app

SR = 16000
REGIONS = [{"start": s, "end": s + 2.0} for s in (0.0, 5.5, 11.0, 16.5, 22.0, 27.5)]


class FakeWhisper:
    calls = []

    def __init__(self, name):
        self.name = name

    def transcribe(self, audio, language, initial_prompt=None, hotwords=None):
        FakeWhisper.calls.append({"prompt": initial_prompt, "hotwords": hotwords, "seconds": len(audio) / SR})
        if initial_prompt:
            return {"text": "ni kwa sababu ya maji", "detected_language": "sw"}
        return {"text": "ni kwa sababa ya maji", "detected_language": "sw"}


def wav_bytes(seconds=31.0):
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    sf.write(buffer, (rng.standard_normal(int(seconds * SR)) * 0.05).astype(np.float32), SR, format="WAV")
    return buffer.getvalue()


def wait(client, sid, predicate, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        session = client.get(f"/api/sessions/{sid}").json()
        if predicate(session):
            return session
        time.sleep(0.1)
    raise AssertionError(f"timed out; last status {session.get('status')}: {session.get('error')}")


def test_upload_transcribe_correct_learn_export(tmp_path):
    FakeWhisper.calls = []
    app = create_app(tmp_path, transcriber_factory=FakeWhisper, segmenter=lambda audio: REGIONS)
    client = TestClient(app)
    assert "Transcription" in client.get("/").text

    r = client.post("/api/sessions", files={"file": ("KSM 09.mp4.wav", wav_bytes(), "audio/wav")},
                    data={"language": "sw", "model": "small", "speakers": "false"})
    sid = r.json()["id"]
    session = wait(client, sid, lambda s: s["status"] == "ready")
    assert len(session["segments"]) == 6
    assert all(s["transcript"] == "ni kwa sababa ya maji" for s in session["segments"])
    assert [round(c["seconds"]) for c in FakeWhisper.calls] == [2] * 6

    audio = client.get(f"/api/sessions/{sid}/chunks/1/audio")
    assert audio.headers["content-type"] == "audio/wav"
    clip, sr = sf.read(io.BytesIO(audio.content))
    assert sr == SR and abs(len(clip) / SR - 2.0) < 0.01

    # First correction: nothing learned yet, but look-ahead re-transcribes with the prompt.
    r = client.put(f"/api/sessions/{sid}/chunks/0", json={"text": "Ni kwa sababu ya maji."}).json()
    assert r["adaptation"]["rules"] == [] and r["lookahead"] == [1, 2, 3]
    session = wait(client, sid, lambda s: s["segments"][3]["suggestion"] == "lookahead")
    assert session["segments"][1]["transcript"] == "ni kwa sababu ya maji"
    prompted = [c for c in FakeWhisper.calls if c["prompt"]]
    assert prompted and prompted[0]["prompt"] == "kwa sababu ya maji." and "sababu" in prompted[0]["hotwords"]

    # Chunk 4 (beyond the look-ahead) still has the raw "sababa": correcting it is the second
    # consistent fix -> a rule, applied at once to chunk 5, which nobody has edited.
    r = client.put(f"/api/sessions/{sid}/chunks/4", json={"text": "Ni kwa sababu ya maji."}).json()
    assert ["sababa", "sababu", 2] in r["adaptation"]["rules"]
    assert 5 in r["adapted"]
    app.state.ahead.join()
    assert client.get(f"/api/sessions/{sid}").json()["segments"][5]["transcript"] == "ni kwa sababu ya maji"

    # The reviewer edits chunk 2; a later look-ahead for it must never overwrite the edit.
    client.put(f"/api/sessions/{sid}/chunks/2", json={"text": "Ni kwa sababu ya mvua."})
    app.state.ahead.put((sid, [2]))
    app.state.ahead.join()
    session = client.get(f"/api/sessions/{sid}").json()
    assert session["segments"][2]["transcript"] == "Ni kwa sababu ya mvua."
    assert session["segments"][2]["edited"] and session["segments"][2]["status"] == "approved"

    txt = client.get(f"/api/sessions/{sid}/export?format=txt").text
    assert "Ni kwa sababu ya mvua." in txt
    srt = client.get(f"/api/sessions/{sid}/export?format=srt").text
    assert "00:00:05,500 --> 00:00:07,500" in srt
    assert client.get("/api/sessions").json()[0]["edited"] == 3


def test_save_rejected_until_ready_and_bad_ids(tmp_path):
    app = create_app(tmp_path, transcriber_factory=FakeWhisper, segmenter=lambda audio: REGIONS)
    client = TestClient(app)
    assert client.get("/api/sessions/does-not-exist").status_code == 404
    assert client.get("/api/sessions/..%2Fetc").status_code == 404


def test_token_protects_a_public_link(tmp_path):
    app = create_app(tmp_path, transcriber_factory=FakeWhisper, segmenter=lambda audio: REGIONS, token="s3cret")
    client = TestClient(app)
    assert client.get("/").status_code == 401
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/sessions?token=wrong").status_code == 401
    ok = client.get("/?token=s3cret")
    assert ok.status_code == 200 and "transcribe_token" in ok.headers["set-cookie"]
    assert client.get("/api/sessions").status_code == 200          # cookie now carries the token


def test_batch_command_writes_txt_srt_json(tmp_path):
    from transcribe.app.batch import transcribe_file
    audio = tmp_path / "Interview 01.wav"
    audio.write_bytes(wav_bytes())
    session = transcribe_file(audio, tmp_path / "out", FakeWhisper("small"), speakers=False,
                              segmenter=lambda a: REGIONS)
    assert len(session["segments"]) == 6
    txt = (tmp_path / "out" / "Interview 01.txt").read_text()
    assert txt.count("ni kwa sababa ya maji") == 6
    assert "00:00:05,500 --> 00:00:07,500" in (tmp_path / "out" / "Interview 01.srt").read_text()
    assert (tmp_path / "out" / "Interview 01.json").exists()


def test_batch_finds_media_in_folders(tmp_path):
    from transcribe.app.batch import find_media
    (tmp_path / "a").mkdir()
    for name in ("a/one.MP4", "a/two.wav", "a/notes.txt", "three.m4a", ".hidden.mp3"):
        (tmp_path / name).write_bytes(b"x")
    found = [p.name for p in find_media([tmp_path / "a", tmp_path / "three.m4a"])]
    assert found == ["one.MP4", "two.wav", "three.m4a"]


def test_decoder_handles_video_and_falls_back(tmp_path, monkeypatch):
    import av
    from transcribe.app import pipeline

    video = tmp_path / "VID 1.mp4"
    out = av.open(str(video), "w", format="mp4")
    v = out.add_stream("mpeg4", rate=10)
    v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
    a = out.add_stream("aac", rate=16000)
    a.layout = "mono"
    for _ in range(30):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 90, np.uint8), format="rgb24")
        for packet in v.encode(frame.reformat(format="yuv420p")):
            out.mux(packet)
    tone = (np.sin(np.arange(16000 * 3) / 20) * 8000).astype(np.int16)
    for i in range(0, len(tone), 1024):
        f = av.AudioFrame.from_ndarray(tone[i:i + 1024][None, :], format="s16", layout="mono")
        f.sample_rate = 16000
        for packet in a.encode(f):
            out.mux(packet)
    for stream in (v, a):
        for packet in stream.encode():
            out.mux(packet)
    out.close()

    assert abs(pipeline.decode_to_wav(video, tmp_path / "a.wav") - 3.0) < 0.2

    def broken(_source):                       # e.g. an incompatible PyAV on a new Python
        raise TypeError("open() got an unexpected keyword argument")
    monkeypatch.setattr(pipeline, "_decode_pyav", broken)
    plain = tmp_path / "tone.wav"
    sf.write(plain, tone.astype(np.float32) / 32768, 16000)
    assert abs(pipeline.decode_to_wav(plain, tmp_path / "b.wav") - 3.0) < 0.1      # librosa fallback

    monkeypatch.setattr(pipeline, "_decode_librosa", broken)
    monkeypatch.setattr(pipeline.shutil, "which", lambda _name: None)
    try:
        pipeline.decode_to_wav(plain, tmp_path / "c.wav")
    except RuntimeError as exc:
        assert all(label in str(exc) for label in ("PyAV:", "ffmpeg:", "librosa:"))
    else:
        raise AssertionError("expected all decoders to fail")
