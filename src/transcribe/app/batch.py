"""Transcribe audio or video files with no web page: writes TXT, SRT and JSON next to each other.

    python -m transcribe.app.batch interview.mp4 --out results
    python -m transcribe.app.batch recordings/ --model large-v3-turbo --language sw   # a whole folder

A folder is searched for audio and video files; files that already have a .txt in --out are skipped,
so a stopped run can be started again.

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


MEDIA = {".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".mov", ".webm", ".avi", ".ogg", ".opus", ".flac", ".aac",
         ".wma", ".3gp", ".mpeg", ".mpg", ".m4v"}


def find_media(paths) -> list:
    """Files as given, plus audio/video files found (recursively) inside any folder."""
    found = []
    for name in paths:
        path = Path(name)
        if path.is_dir():
            found += sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA
                            and not p.name.startswith("."))
        else:
            found.append(path)
    return found


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
    files = find_media(args.audio)
    if not files:
        raise SystemExit("No audio or video files found in: " + ", ".join(args.audio))
    for path in files:
        if (out / f"{safe_name(path.stem)}.txt").exists():
            print(f"{path.name}: already done, skipped", flush=True)
            continue
        print(f"{path.name}: starting ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
        started = time.time()
        session = transcribe_file(path, out, runner, args.language, args.model, not args.no_speakers)
        print(f"{path.name}: {len(session['segments'])} chunks, "
              f"{(session['duration'] or 0) / 60:.1f} min of audio in {(time.time() - started) / 60:.1f} min "
              f"-> {out / safe_name(path.stem)}.txt / .srt / .json", flush=True)


if __name__ == "__main__":
    main()
