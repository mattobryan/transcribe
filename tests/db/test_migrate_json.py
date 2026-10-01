"""P0.7: importing JSON review manifests."""
import json
import subprocess
import sys

from sqlalchemy import func, select

from transcribe.db import create_engine_for, session_factory
from transcribe.db.models import Recording, Segment, SegmentRevision, TranscriptTurn


def write_manifest(path, audio):
    audio.write_bytes(b"RIFF fake audio")
    path.write_text(json.dumps({
        "schema_version": 1, "recording_id": "KSM09", "source_audio": str(audio), "origin": "pilot_alignment",
        "transcript_turns": [{"turn_id": "p0001", "speaker_id": "interviewer", "text": "Habari yako"},
                             {"turn_id": "p0002", "speaker_id": "respondent", "text": "Nzuri sana"}],
        "segments": [
            {"segment_id": "a", "start": 1.0, "end": 3.0, "transcript": "Habari yako", "status": "approved",
             "speaker_id": "interviewer", "alignment_conf": 0.9, "flags": []},
            {"segment_id": "b", "start": 3.5, "end": 5.0, "transcript": "Nzuri sana", "status": "approved",
             "speaker_id": "respondent", "unclear": True, "flags": ["low_alignment"]},
            {"segment_id": "c", "start": 6.0, "end": 8.0, "transcript": "candidate text", "status": "candidate"},
        ]}))


def run(args):
    return subprocess.run([sys.executable, "-m", "transcribe.db.migrate_json", *args],
                          capture_output=True, text=True, check=True).stdout


def test_import_keeps_approved_and_is_idempotent(tmp_path):
    manifest, db = tmp_path / "review_manifest.json", tmp_path / "t.db"
    write_manifest(manifest, tmp_path / "ksm09.wav")
    out = run(["--db", str(db), str(manifest)])
    assert "2 approved segments kept, 1 candidates dropped, 2 transcript turns" in out
    assert "already imported" in run(["--db", str(db), str(manifest)])

    engine = create_engine_for(str(db))
    with session_factory(engine)() as s:
        assert s.scalar(select(func.count()).select_from(Recording)) == 1
        assert s.scalar(select(func.count()).select_from(TranscriptTurn)) == 2
        segments = s.scalars(select(Segment).order_by(Segment.ordinal)).all()
        assert [(x.origin, x.status, x.start_s) for x in segments] == \
            [("alignment", "approved", 1.0), ("alignment", "approved", 3.5)]
        revision = s.get(SegmentRevision, segments[1].current_revision_id)
        assert (revision.text, revision.source, json.loads(revision.flags)) == \
            ("Nzuri sana", "human", {"unclear": True})
    engine.dispose()
