"""Transcribe an audio or video file with the ElevenLabs Scribe API (no GPU needed).

    export ELEVENLABS_API_KEY=...            # never paste the key into a chat or commit it
    python scripts/elevenlabs_transcribe.py interview.mp4 --out results

Writes <name>.txt (speaker turns with times), <name>.srt and <name>.elevenlabs.json (raw reply).
Add --language sw to force a language; the default lets Scribe detect it.
"""

import argparse
import json
import os
import sys
from pathlib import Path

URL = "https://api.elevenlabs.io/v1/speech-to-text"


def ts(seconds: float, comma: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, m, s, r = ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{r:03d}" if comma else f"{h}:{m:02d}:{s:02d}"


def turns(reply: dict, max_gap: float = 1.5, max_len: float = 30.0) -> list:
    """Words -> chunks that break on a speaker change, a pause, or after max_len seconds."""
    out, cur = [], None
    for w in reply.get("words", []):
        if w.get("type") == "audio_event":
            continue
        speaker = w.get("speaker_id")
        if (cur is None or speaker != cur["speaker"] or w["start"] - cur["end"] > max_gap
                or w["end"] - cur["start"] > max_len) and w.get("type") == "word":
            cur = {"speaker": speaker, "start": w["start"], "end": w["end"], "text": ""}
            out.append(cur)
        if cur is None:
            continue
        cur["text"] += w["text"]
        if w.get("type") == "word":
            cur["end"] = w["end"]
    for c in out:
        c["text"] = " ".join(c["text"].split())
    return [c for c in out if c["text"]]


def render(reply: dict):
    items = turns(reply)
    txt = "\n".join(f"[{ts(c['start'])}] {c['speaker'] + ': ' if c['speaker'] else ''}{c['text']}" for c in items)
    srt = "\n".join(f"{i + 1}\n{ts(c['start'], True)} --> {ts(c['end'], True)}\n"
                    f"{c['speaker'] + ': ' if c['speaker'] else ''}{c['text']}\n" for i, c in enumerate(items))
    return txt + "\n", srt


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio")
    parser.add_argument("--out", default="results")
    parser.add_argument("--model", default="scribe_v2", help="scribe_v2 (default); scribe_v1 if v2 is not offered. Not the medical model.")
    parser.add_argument("--keyterms", default=None, help="a text file with one word or phrase per line (about $0.05 per hour extra)")
    parser.add_argument("--language", default=None, help="e.g. sw or en; omit to auto-detect")
    parser.add_argument("--no-speakers", action="store_true")
    args = parser.parse_args(argv)

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("Set ELEVENLABS_API_KEY first (elevenlabs.io -> Developers -> API keys).")
    import requests

    path = Path(args.audio)
    data = {"model_id": args.model, "diarize": str(not args.no_speakers).lower(),
            "timestamps_granularity": "word", "tag_audio_events": "false"}
    if args.language:
        data["language_code"] = args.language
    # the settings used for the Kenyan interviews: diarize true, word timestamps, audio events off, language left empty
    pairs = list(data.items())
    if args.keyterms:
        pairs += [("keyterms", line.strip()) for line in Path(args.keyterms).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Uploading {path.name} ({path.stat().st_size / 1e6:.0f} MB); this takes a few minutes ...", flush=True)
    with path.open("rb") as handle:
        response = requests.post(URL, headers={"xi-api-key": key}, data=pairs,
                                 files={"file": (path.name, handle)}, timeout=3600)
    if response.status_code != 200:
        sys.exit(f"ElevenLabs error {response.status_code}: {response.text[:500]}")
    reply = response.json()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    txt, srt = render(reply)
    (out / f"{path.stem}.elevenlabs.json").write_text(json.dumps(reply, ensure_ascii=False), encoding="utf-8")
    (out / f"{path.stem}.txt").write_text(txt, encoding="utf-8")
    (out / f"{path.stem}.srt").write_text(srt, encoding="utf-8")
    print(f"Done: detected language {reply.get('language_code')}, written to {out}/{path.stem}.txt / .srt")


if __name__ == "__main__":
    main()
