"""Database schema and invariants (P0.6): migration, invariants 1, 2 and 8, constraints."""
import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from transcribe.db import create_engine_for, migrate, session_factory
from transcribe.db.models import Base, ModelVersion, Recording, Segment, SegmentRevision
from transcribe.db.service import AppendOnlyViolation, RevisionConflict, add_revision


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "transcribe.db")
    migrate(path)
    engine = create_engine_for(path)
    Session = session_factory(engine)
    with Session() as session:
        yield session
    engine.dispose()


def make_segment(session):
    recording = Recording(title="KSM09", original_path="o.mp3", sha256="abc", duration_s=7525.0,
                          consent_scope="training")
    session.add(recording)
    session.flush()
    segment = Segment(recording_id=recording.id, origin="alignment", ordinal=0, start_s=1.0, end_s=4.0)
    session.add(segment)
    session.flush()
    return segment


def test_migration_creates_every_table(tmp_path):
    path = str(tmp_path / "t.db")
    migrate(path)
    engine = create_engine_for(path)
    tables = set(inspect(engine).get_table_names()) - {"alembic_version"}
    assert tables == set(Base.metadata.tables)
    indexes = {i["name"] for i in inspect(engine).get_indexes("segments")}
    assert "ix_segments_queue" in indexes
    engine.dispose()


def test_revision_moves_segment_and_rejects_stale_base(db):
    segment = make_segment(db)
    r1 = add_revision(db, segment, text="habari yako", start_s=1.0, end_s=4.0, status="candidate",
                      source="aligned_transcript", author="aligner")
    r2 = add_revision(db, segment, text="habari yako sana", start_s=1.1, end_s=4.2, status="approved",
                      source="human", author="matt", base_revision_id=r1.id)
    assert (segment.current_revision_id, segment.start_s, segment.end_s, segment.status) == \
        (r2.id, 1.1, 4.2, "approved")
    assert (r2.revision_no, r2.parent_revision_id) == (2, r1.id)
    with pytest.raises(RevisionConflict):
        add_revision(db, segment, text="other tab", start_s=1.0, end_s=4.0, status="draft",
                     source="human", author="matt", base_revision_id=r1.id)


def test_revisions_are_append_only(db):
    segment = make_segment(db)
    revision = add_revision(db, segment, text="sawa", start_s=1.0, end_s=4.0, status="candidate",
                            source="aligned_transcript", author="aligner")
    db.commit()
    revision.text = "changed"
    with pytest.raises(AppendOnlyViolation):
        db.flush()
    db.rollback()
    db.delete(db.get(SegmentRevision, revision.id))
    with pytest.raises(AppendOnlyViolation):
        db.flush()
    db.rollback()


def test_only_one_production_model(db):
    db.add(ModelVersion(name="whisper-small-kct-v1", family="whisper", base_model="openai/whisper-small",
                        hf_path="m1", status="production"))
    db.add(ModelVersion(name="whisper-small-kct-v2", family="whisper", base_model="openai/whisper-small",
                        hf_path="m2", status="candidate"))
    db.flush()
    db.add(ModelVersion(name="whisper-small-kct-v3", family="whisper", base_model="openai/whisper-small",
                        hf_path="m3", status="production"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_check_constraints(db):
    db.add(Recording(title="x", original_path="x", sha256="x1", duration_s=1.0, consent_scope="everything"))
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()
    segment = make_segment(db)
    segment.end_s = 0.5                                    # end before start
    with pytest.raises(IntegrityError):
        db.flush()
