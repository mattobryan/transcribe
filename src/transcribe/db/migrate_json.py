"""Import existing JSON review manifests into the database (Implementation Plan P0.7).

Reads the old chunk manifests (``data/projects/<name>/audio.json``) and pilot
review manifests (``data/pilot/<name>/review_manifest.json``). Approved segments
keep their reviewed text as revision 1 (source ``human``); candidate segments
are dropped and counted, because re-running alignment produces better ones.
Re-importing the same recording is skipped.

    python -m transcribe.db.migrate_json --db data/transcribe.db data/projects/*/audio.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

from sqlalchemy import select

from . import create_engine_for, migrate, session_factory
from .models import Recording, Segment, SourceTranscript, TranscriptTurn
from .service import add_revision

APPROVED = {"approved", "approved_eval_only"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _recording_hash(manifest: Dict, manifest_path: Path) -> str:
    audio = Path(manifest.get("source_audio") or "")
    if audio.is_file():
        return _sha256(audio)
    # Audio not on this machine: a stable stand-in so re-imports are still skipped.
    return "missing:" + hashlib.sha256(f"{manifest.get('recording_id')}|{audio}".encode()).hexdigest()


def import_manifest(session, manifest_path: Path, consent_scope: str = "training") -> Dict[str, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments: List[Dict] = manifest.get("segments", [])
    sha = _recording_hash(manifest, manifest_path)
    if session.scalar(select(Recording).where(Recording.sha256 == sha)):
        return {"skipped": 1, "approved": 0, "dropped": 0, "turns": 0}
    duration = max([float(s.get("source_duration") or 0) for s in segments]
                   + [float(s.get("end") or 0) for s in segments] + [0.0])
    recording = Recording(
        title=manifest.get("recording_id") or manifest_path.parent.name,
        original_path=str(manifest.get("source_audio") or ""), sha256=sha, duration_s=duration,
        consent_scope=consent_scope, status="ready_for_review",
        notes=f"imported from {manifest_path}")
    session.add(recording)
    session.flush()

    turn_ids: Dict[str, str] = {}
    turns = manifest.get("transcript_turns") or []
    if turns:
        source = SourceTranscript(recording_id=recording.id, path=str(manifest_path), format="txt",
                                  sha256=_sha256(manifest_path), parser="migrate_json",
                                  parser_version="1")
        session.add(source)
        session.flush()
        for ordinal, turn in enumerate(turns):
            text = turn.get("text", "")
            row = TranscriptTurn(source_transcript_id=source.id, ordinal=ordinal,
                                 speaker_label=turn.get("speaker_id"), text_raw=text,
                                 text_align=text.lower(), word_map="[]",
                                 spans=json.dumps(turn.get("spans", [])),
                                 annotations=json.dumps(turn.get("annotations", [])))
            session.add(row)
            session.flush()
            if turn.get("turn_id"):
                turn_ids[turn["turn_id"]] = row.id

    origin = "alignment" if manifest.get("origin") == "pilot_alignment" else "vad"
    approved = dropped = 0
    for ordinal, item in enumerate(segments):
        status = item.get("status", "candidate")
        text = (item.get("transcript") or "").strip()
        start, end = float(item.get("start", 0.0)), float(item.get("end", 0.0))
        if status not in APPROVED or not text or end <= start:
            dropped += 1
            continue
        flags = {name: bool(item.get(name)) for name in ("unclear", "overlap") if name in item}
        segment = Segment(recording_id=recording.id, origin=origin, ordinal=ordinal, start_s=start, end_s=end,
                          speaker_label=item.get("speaker_id"), alignment_conf=item.get("alignment_conf"),
                          auto_flags=json.dumps(item.get("flags", [])))
        session.add(segment)
        session.flush()
        add_revision(session, segment, text=text, start_s=start, end_s=end, status=status,
                     source="human", author="migrate_json", flags=flags, speaker_label=item.get("speaker_id"))
        approved += 1
    return {"skipped": 0, "approved": approved, "dropped": dropped, "turns": len(turns)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--db", default="data/transcribe.db")
    parser.add_argument("--consent", default="training", choices=["training", "eval_only", "transcription_only"])
    args = parser.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    migrate(args.db)
    engine = create_engine_for(args.db)
    Session = session_factory(engine)
    totals = {"skipped": 0, "approved": 0, "dropped": 0, "turns": 0}
    with Session() as session:
        for name in args.manifests:
            path = Path(name)
            if not path.is_file():
                print(f"not found: {path}", file=sys.stderr)
                continue
            counts = import_manifest(session, path, args.consent)
            session.commit()
            for key in totals:
                totals[key] += counts[key]
            state = "already imported, skipped" if counts["skipped"] else (
                f"{counts['approved']} approved segments kept, {counts['dropped']} candidates dropped, "
                f"{counts['turns']} transcript turns")
            print(f"{path}: {state}")
    print(f"Total: {totals['approved']} approved kept, {totals['dropped']} candidates dropped, "
          f"{totals['skipped']} manifests skipped. Database: {args.db}")


if __name__ == "__main__":
    main()
