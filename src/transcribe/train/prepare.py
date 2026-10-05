"""Build the training set from the corrector's exports.

    python -m transcribe.train.prepare recordings.json --out data/train

``recordings.json``::

    {"recordings": [
       {"name": "Interview 1", "audio": "interview1.mp4", "log": "Interview 1 corrections log.json",
        "labels": "Interview 1 language labels.csv", "split": "dev"},
       {"name": "Client A", "audio": "a.mp3", "log": "...", "labels": "...", "split": "train"},
       {"name": "Client C", "audio": "c.mp3", "log": "...", "labels": "...", "split": "test"}]}

Give every recording a split; a recording is never divided. Writes ``train.jsonl`` / ``dev.jsonl`` / ``test.jsonl`` and the clips.
"""

import argparse
import json
import tempfile
from pathlib import Path

from transcribe.app.pipeline import decode_to_wav

from .dataset import SPLITS, build_rows, slug, write_manifests


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--out", default="data/train")
    parser.add_argument("--all-chunks", action="store_true", help="include chunks nobody marked as checked")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base = Path(args.config).resolve().parent
    for rec in config["recordings"]:                      # paths in the file are relative to the file itself
        for key in ("audio", "log", "labels"):
            if rec.get(key) and not Path(rec[key]).is_absolute():
                rec[key] = str(base / rec[key])
    out = Path(args.out)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for rec in config["recordings"]:
            if rec.get("split", "train") not in SPLITS:
                raise SystemExit(f"{rec['name']}: split must be one of {SPLITS}")
            wav = Path(tmp) / (slug(rec["name"]) + ".wav")
            seconds = decode_to_wav(Path(rec["audio"]), wav)
            part = build_rows(rec, wav, out / "clips" / slug(rec["name"]), only_checked=not args.all_chunks, base=out)
            hours = sum(r["duration"] for r in part) / 3600
            print(f"{rec['name']}: {len(part)} chunks, {hours:.2f} h of {seconds / 3600:.2f} h, split {rec.get('split', 'train')}", flush=True)
            rows += part
            wav.unlink()
    counts = write_manifests(rows, out)
    print("manifests:", counts, "->", out)


if __name__ == "__main__":
    main()
