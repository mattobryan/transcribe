"""Read and write versioned preprocessing manifests."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .schema import Manifest


def save_manifest(manifest: Manifest, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_manifest(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_training_manifest(manifest: Manifest, path: str) -> None:
    """Export approved segment records for the existing trainer contract."""
    records = [
        {"audio_file": segment["audio_file"], "transcript": segment["transcript"]}
        for segment in manifest.segments
        if segment.get("status") == "approved"
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def export_segments_manifest(source_path: str, output_path: str) -> int:
    """Project approved rich segment records to the trainer's simple format."""
    source = load_manifest(source_path)
    records = [
        {"audio_file": segment["audio_file"], "transcript": segment["transcript"]}
        for segment in source.get("segments", [])
        if segment.get("status") == "approved"
        and segment.get("audio_file")
        and segment.get("transcript", "").strip()
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)