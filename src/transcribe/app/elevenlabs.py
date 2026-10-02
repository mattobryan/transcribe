"""Open an ElevenLabs Scribe transcript (the JSON from the API or the Speech to Text page) in the
correction app: words are grouped into review chunks and paired with the audio or video file.

    python -m transcribe.app --open-elevenlabs transcript.json --audio interview.mp4
"""

import json
from pathlib import Path
from typing import Dict, List

from . import pipeline
from .store import Store

MIN_CHUNK_S = 6.0        # a speaker change closes a chunk only after this long
TARGET_CHUNK_S = 15.0    # a pause closes a chunk after this long
MAX_CHUNK_S = 30.0
PAUSE_S = 0.8
BAD_WORD_LOGPROB = -0.5  # a word Scribe was less sure of (its scores are high overall)
BAD_WORD_SHARE = 0.1     # chunks with this share of such words are flagged


def chunks(reply: Dict) -> List[Dict]:
    """Review chunks (start, end, text, speaker, share of unsure words) from a Scribe reply."""
    words = [w for w in reply.get("words", []) if w.get("type") in ("word", "spacing")]
    names: Dict[str, str] = {}
    out: List[Dict] = []
    cur = None

    def close():
        if cur:
            weights = cur.pop("weights")
            cur["speaker"] = max(weights, key=weights.get) if weights else None
            cur["text"] = " ".join(cur["text"].split())
            if cur["text"]:
                out.append(cur)

    for w in words:
        if w["type"] == "spacing":
            if cur:
                cur["text"] += w["text"]
            continue
        speaker = names.setdefault(w.get("speaker_id"), f"Speaker {len(names) + 1}") if w.get("speaker_id") else None
        if cur is not None:
            length = cur["end"] - cur["start"]
            majority = max(cur["weights"], key=cur["weights"].get) if cur["weights"] else None
            if (length + (w["end"] - cur["end"]) > MAX_CHUNK_S
                    or (speaker != majority and length >= MIN_CHUNK_S and w["start"] - cur["end"] >= 0.2)
                    or (w["start"] - cur["end"] >= PAUSE_S and length >= TARGET_CHUNK_S)):
                close()
                cur = None
        if cur is None:
            cur = {"start": w["start"], "end": w["end"], "text": "", "weights": {}, "bad": 0, "words": 0}
        cur["text"] += w["text"]
        cur["end"] = w["end"]
        cur["words"] += 1
        cur["bad"] += w.get("logprob", 0.0) < BAD_WORD_LOGPROB
        if speaker:
            cur["weights"][speaker] = cur["weights"].get(speaker, 0.0) + (w["end"] - w["start"])
    close()
    for c in out:
        c["unsure"] = c.pop("bad") / max(1, c.pop("words"))
    return out


def import_elevenlabs(store: Store, json_path: Path, audio_path: Path) -> str:
    reply = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if "words" not in reply:
        raise ValueError("This file has no word list. In the playground set timestamps_granularity to word.")
    session = store.create(title=Path(audio_path).stem, source_audio=str(audio_path),
                           settings={"language": "auto", "model": "elevenlabs", "speakers": True})
    folder = store.folder(session["id"])
    duration = pipeline.decode_to_wav(Path(audio_path), folder / "audio16k.wav")
    items = chunks(reply)
    session["segments"] = [{
        "segment_id": f"{session['id']}_{i:05d}", "sequence": i, "start": round(c["start"], 3),
        "end": round(min(duration, c["end"]), 3), "duration": round(c["end"] - c["start"], 3),
        "transcript": c["text"], "asr_hypothesis": c["text"], "suggestion": "model", "status": "candidate",
        "edited": False, "speaker_id": c["speaker"],
        **({"flag": ["unsure-words"]} if c["unsure"] >= BAD_WORD_SHARE else {}),
    } for i, c in enumerate(items)]
    session.update(status="ready", duration=round(duration, 2), speakers=True,
                   progress={"stage": "Done", "done": len(items), "total": len(items)})
    store.save(session)
    return session["id"]
