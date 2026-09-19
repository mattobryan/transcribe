"""Detect speech and export candidate chunks for later transcript review."""

import argparse
import json
from pathlib import Path

from src.transcribe.preprocessing.chunker import export_chunks
from src.transcribe.preprocessing.manifest import load_manifest
from src.transcribe.preprocessing.vad import detect_speech


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect speech and export audio chunks")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--transcript-manifest")
    parser.add_argument("--recording-id")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--threshold-db", type=float, default=-35.0)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    recording_id = args.recording_id or audio_path.stem
    intervals = detect_speech(str(audio_path), threshold_db=args.threshold_db)
    segments = export_chunks(
        str(audio_path), intervals, args.output_dir, recording_id,
        max_seconds=args.max_seconds,
    )
    transcript_turns = []
    if args.transcript_manifest:
        transcript_turns = load_manifest(args.transcript_manifest).get("transcript_turns", [])
    output = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 1,
        "recording_id": recording_id,
        "source_audio": str(audio_path),
        "vad": {"backend": "energy", "threshold_db": args.threshold_db},
        "speech_intervals": [interval.to_dict() for interval in intervals],
        "transcript_turns": transcript_turns,
        "segments": segments,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Detected {len(intervals)} speech intervals")
    print(f"Exported {len(segments)} candidate chunks")
    print(f"Saved manifest to {output}")


if __name__ == "__main__":
    main()