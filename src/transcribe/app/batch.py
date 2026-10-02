"""Transcribe audio files with no web page: writes TXT, SRT and JSON next to each other.

    python -m transcribe.app.batch interview.mp3 --out results
    python -m transcribe.app.batch *.mp3 --model large-v3-turbo --language sw

The ``<name>/session.json`` it writes opens in the correction app later:
``python -m transcribe.app --open results/<name>/session.json``.
"""

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

from . import pipeline
from .server import render_export
from .store import Store, safe_name


def transcribe_file(audio: Path, out: Path, transcriber, language: str = "sw", model: str = "small",
                    speakers: bool = True, segmenter=None) -> Dict:
    store = Store(out)
    session = store.create(title=audio.stem, source_audio="", settings={
        "language": language, "model": model, "speakers": speakers, "filename": audio.name})
    folder = store.folder(session["id"])
    target = folder / f"original{audio.suffix.lower() or '.bin'}"
    shutil.copyfile(audio, target)
    session["source_audio"] = str(target)
    store.save(session)
    kwargs = {"segmenter": segmenter} if segmenter else {}
    session = pipeline.transcribe_session(session, folder, transcriber, store.save, **kwargs)
    name = safe_name(audio.stem)
    for fmt in ("txt", "srt", "json"):
        body, _ = render_export(session, fmt)
        (out / f"{name}.{fmt}").write_text(body, encoding="utf-8")
    return session


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--out", default="results")
    parser.add_argument("--language", default="sw", choices=["sw", "en", "auto"])
    parser.add_argument("--model", default="small", help="small, large-v3-turbo, ...")
    parser.add_argument("--no-speakers", action="store_true")
    args = parser.parse_args(argv)

    from transcribe.pilot.asr import WhisperRunner
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = WhisperRunner(args.model)
    print(f"Whisper {args.model} on {runner.device}", flush=True)
    for name in args.audio:
        path = Path(name)
        started = time.time()
        session = transcribe_file(path, out, runner, args.language, args.model, not args.no_speakers)
        print(f"{path.name}: {len(session['segments'])} chunks, "
              f"{(session['duration'] or 0) / 60:.1f} min of audio in {(time.time() - started) / 60:.1f} min "
              f"-> {out / safe_name(path.stem)}.txt / .srt / .json", flush=True)


if __name__ == "__main__":
    main()
