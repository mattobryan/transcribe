"""SQLAlchemy models for the schema in docs/pdf/05_BACKEND_SCHEMA.pdf (section 2).

SQLite in WAL mode for v1; types are portable to PostgreSQL (JSON is stored as
TEXT with a json_valid() CHECK on SQLite). Enumerations are TEXT columns with
CHECK constraints. IDs are ULID strings; timestamps are ISO-8601 UTC strings.
"""

import os
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (CheckConstraint, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id() -> str:
    """ULID: 48-bit millisecond time + 80 random bits, Crockford base32 (sortable)."""
    value = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    return "".join(_CROCKFORD[(value >> shift) & 31] for shift in range(125, -1, -5))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _in(column: str, *values: str) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _json(column: str) -> CheckConstraint:
    return CheckConstraint(f"json_valid({column})", name=f"ck_{column}_json")


class Base(DeclarativeBase):
    pass


class Timestamps:
    created_at: Mapped[str] = mapped_column(Text, default=now)


# ───────────────────────── Sources ─────────────────────────
class Recording(Timestamps, Base):
    __tablename__ = "recordings"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(Text)
    original_path: Mapped[str] = mapped_column(Text)
    normalized_path: Mapped[Optional[str]] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(Text, unique=True)
    duration_s: Mapped[float] = mapped_column(Float)
    domain: Mapped[str] = mapped_column(Text, default="interview")
    consent_scope: Mapped[str] = mapped_column(Text)
    recorded_on: Mapped[Optional[str]] = mapped_column(Text)
    region: Mapped[Optional[str]] = mapped_column(Text)        # free text; never ethnicity
    status: Mapped[str] = mapped_column(Text, default="imported")
    notes: Mapped[Optional[str]] = mapped_column(Text)
    updated_at: Mapped[str] = mapped_column(Text, default=now, onupdate=now)
    __table_args__ = (
        CheckConstraint(_in("domain", "interview", "tv", "radio", "church", "meeting", "other"), name="ck_domain"),
        CheckConstraint(_in("consent_scope", "training", "eval_only", "transcription_only"), name="ck_consent"),
        CheckConstraint(_in("status", "imported", "aligning", "ready_for_review", "alignment_suspect",
                            "reviewed", "archived"), name="ck_status"),
    )


class SourceTranscript(Timestamps, Base):
    __tablename__ = "source_transcripts"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"))
    path: Mapped[str] = mapped_column(Text)
    format: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(Text)
    parser: Mapped[str] = mapped_column(Text)
    parser_version: Mapped[str] = mapped_column(Text)
    italic_means: Mapped[Optional[str]] = mapped_column(Text, default="sw")
    __table_args__ = (
        UniqueConstraint("recording_id", "sha256"),
        CheckConstraint(_in("format", "docx", "pdf", "txt"), name="ck_format"),
        CheckConstraint(_in("italic_means", "sw", "en", "none"), name="ck_italic"),
    )


class TranscriptTurn(Base):
    __tablename__ = "transcript_turns"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    source_transcript_id: Mapped[str] = mapped_column(ForeignKey("source_transcripts.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    speaker_label: Mapped[Optional[str]] = mapped_column(Text)
    speaker_role: Mapped[Optional[str]] = mapped_column(Text)
    text_raw: Mapped[str] = mapped_column(Text)
    text_align: Mapped[str] = mapped_column(Text)
    word_map: Mapped[str] = mapped_column(Text)
    spans: Mapped[str] = mapped_column(Text, default="[]")
    annotations: Mapped[str] = mapped_column(Text, default="[]")
    __table_args__ = (
        UniqueConstraint("source_transcript_id", "ordinal"),
        CheckConstraint(_in("speaker_role", "interviewer", "respondent", "host", "preacher", "audience", "other"),
                        name="ck_role"),
        _json("word_map"), _json("spans"), _json("annotations"),
    )


# ───────────────────────── Alignment ─────────────────────────
class AlignmentRun(Timestamps, Base):
    __tablename__ = "alignment_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"))
    source_transcript_id: Mapped[str] = mapped_column(ForeignKey("source_transcripts.id"))
    aligner: Mapped[str] = mapped_column(Text)
    aligner_version: Mapped[str] = mapped_column(Text)
    params: Mapped[str] = mapped_column(Text)
    mean_confidence: Mapped[Optional[float]] = mapped_column(Float)
    status: Mapped[str] = mapped_column(Text)
    finished_at: Mapped[Optional[str]] = mapped_column(Text)
    __table_args__ = (CheckConstraint(_in("status", "running", "done", "failed"), name="ck_status"),
                      _json("params"))


class AlignedWord(Base):
    __tablename__ = "aligned_words"
    alignment_run_id: Mapped[str] = mapped_column(ForeignKey("alignment_runs.id", ondelete="CASCADE"),
                                                  primary_key=True)
    turn_id: Mapped[str] = mapped_column(ForeignKey("transcript_turns.id"), primary_key=True)
    word_idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    word: Mapped[str] = mapped_column(Text)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)


# ───────────────────────── Segments & review ─────────────────────────
SEGMENT_STATUSES = ("candidate", "in_review", "draft", "approved", "approved_eval_only", "rejected")


class Segment(Timestamps, Base):
    __tablename__ = "segments"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id", ondelete="CASCADE"))
    alignment_run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("alignment_runs.id"))
    origin: Mapped[str] = mapped_column(Text)
    ordinal: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)          # mirrors the current revision
    end_s: Mapped[float] = mapped_column(Float)
    speaker_label: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="candidate")
    current_revision_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("segment_revisions.id", use_alter=True, name="fk_segments_current_revision"))
    alignment_conf: Mapped[Optional[float]] = mapped_column(Float)
    auto_flags: Mapped[str] = mapped_column(Text, default="[]")
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    audio_cache_path: Mapped[Optional[str]] = mapped_column(Text)
    updated_at: Mapped[str] = mapped_column(Text, default=now, onupdate=now)
    __table_args__ = (
        UniqueConstraint("recording_id", "ordinal"),
        CheckConstraint("end_s > start_s", name="ck_bounds"),
        CheckConstraint(_in("origin", "alignment", "vad", "inference_correction"), name="ck_origin"),
        CheckConstraint(_in("status", *SEGMENT_STATUSES), name="ck_status"),
        _json("auto_flags"),
        Index("ix_segments_queue", "recording_id", "status", text("priority DESC")),
    )


class SegmentTurn(Base):
    __tablename__ = "segment_turns"
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.id", ondelete="CASCADE"), primary_key=True)
    turn_id: Mapped[str] = mapped_column(ForeignKey("transcript_turns.id"), primary_key=True)


class SegmentRevision(Timestamps, Base):
    """Append-only (invariant 1): rows are never updated or deleted, except by cascade."""
    __tablename__ = "segment_revisions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.id", ondelete="CASCADE"))
    revision_no: Mapped[int] = mapped_column(Integer)
    parent_revision_id: Mapped[Optional[str]] = mapped_column(ForeignKey("segment_revisions.id"))
    text: Mapped[str] = mapped_column(Text)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    speaker_label: Mapped[Optional[str]] = mapped_column(Text)
    flags: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text)
    hypothesis_id: Mapped[Optional[str]] = mapped_column(ForeignKey("hypotheses.id"))
    author: Mapped[str] = mapped_column(Text)
    __table_args__ = (
        UniqueConstraint("segment_id", "revision_no"),
        CheckConstraint(_in("source", "aligned_transcript", "vad_empty", "model_hypothesis", "human"),
                        name="ck_source"),
        CheckConstraint(_in("status", *SEGMENT_STATUSES), name="ck_status"),
        _json("flags"),
    )


class LanguageSpan(Base):
    __tablename__ = "language_spans"
    revision_id: Mapped[str] = mapped_column(ForeignKey("segment_revisions.id", ondelete="CASCADE"),
                                             primary_key=True)
    char_start: Mapped[int] = mapped_column(Integer, primary_key=True)
    char_end: Mapped[int] = mapped_column(Integer)
    lang: Mapped[str] = mapped_column(Text)
    __table_args__ = (CheckConstraint("char_end > char_start", name="ck_span"),
                      CheckConstraint(_in("lang", "en", "sw", "mixed", "luo", "other", "unclear"), name="ck_lang"))


class Hypothesis(Timestamps, Base):
    __tablename__ = "hypotheses"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.id", ondelete="CASCADE"))
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    text: Mapped[str] = mapped_column(Text)
    words: Mapped[Optional[str]] = mapped_column(Text)
    avg_logprob: Mapped[Optional[float]] = mapped_column(Float)
    no_speech_prob: Mapped[Optional[float]] = mapped_column(Float)
    wer_vs_current: Mapped[Optional[float]] = mapped_column(Float)
    __table_args__ = (CheckConstraint("words IS NULL OR json_valid(words)", name="ck_words_json"),)


# ───────────────────────── Datasets, training, models ─────────────────────────
class Dataset(Timestamps, Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(Text, unique=True)
    filter_spec: Mapped[str] = mapped_column(Text)
    split_spec: Mapped[str] = mapped_column(Text)
    test_set_name: Mapped[str] = mapped_column(Text)
    normalizer_version: Mapped[str] = mapped_column(Text)
    sha256: Mapped[Optional[str]] = mapped_column(Text)
    frozen: Mapped[int] = mapped_column(Integer, default=0)
    export_path: Mapped[Optional[str]] = mapped_column(Text)
    stats: Mapped[Optional[str]] = mapped_column(Text)
    __table_args__ = (CheckConstraint("frozen IN (0, 1)", name="ck_frozen"),
                      _json("filter_spec"), _json("split_spec"))


class DatasetItem(Base):
    __tablename__ = "dataset_items"
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.id"), primary_key=True)
    revision_id: Mapped[str] = mapped_column(ForeignKey("segment_revisions.id"))
    split: Mapped[str] = mapped_column(Text)
    __table_args__ = (CheckConstraint(_in("split", "train", "dev", "test"), name="ck_split"),
                      Index("ix_dataset_items_split", "dataset_id", "split"))


class TestSetRecording(Base):
    __tablename__ = "test_set_recordings"
    test_set_name: Mapped[str] = mapped_column(Text, primary_key=True)
    recording_id: Mapped[str] = mapped_column(ForeignKey("recordings.id"), primary_key=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    base_model: Mapped[str] = mapped_column(Text)
    platform: Mapped[str] = mapped_column(Text)
    config: Mapped[str] = mapped_column(Text)
    code_commit: Mapped[Optional[str]] = mapped_column(Text)
    notebook_ref: Mapped[Optional[str]] = mapped_column(Text)
    dev_metrics: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[str]] = mapped_column(Text)
    finished_at: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    __table_args__ = (CheckConstraint(_in("platform", "kaggle", "local", "other"), name="ck_platform"),
                      CheckConstraint(_in("status", "registered", "failed"), name="ck_status"),
                      _json("config"))


class ModelVersion(Timestamps, Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(Text, unique=True)
    family: Mapped[str] = mapped_column(Text)
    base_model: Mapped[str] = mapped_column(Text)
    training_run_id: Mapped[Optional[str]] = mapped_column(ForeignKey("training_runs.id"))
    hf_path: Mapped[str] = mapped_column(Text)
    ct2_path: Mapped[Optional[str]] = mapped_column(Text)
    decode_params: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(Text, default="candidate")
    promoted_at: Mapped[Optional[str]] = mapped_column(Text)
    promotion_note: Mapped[Optional[str]] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint(_in("family", "whisper", "w2vbert_ctc", "legacy_ctc"), name="ck_family"),
        CheckConstraint(_in("status", "converting", "candidate", "production", "retired", "broken"),
                        name="ck_status"),
        _json("decode_params"),
        # Invariant 8: at most one production model.
        Index("ux_one_production", "status", unique=True, sqlite_where=text("status = 'production'"),
              postgresql_where=text("status = 'production'")),
    )


class EvalRun(Timestamps, Base):
    __tablename__ = "eval_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    split: Mapped[str] = mapped_column(Text)
    normalizer_version: Mapped[str] = mapped_column(Text)
    metrics: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    __table_args__ = (CheckConstraint(_in("split", "dev", "test", "external"), name="ck_split"),
                      CheckConstraint(_in("status", "queued", "running", "done", "failed"), name="ck_status"))


class EvalResult(Base):
    __tablename__ = "eval_results"
    eval_run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), primary_key=True)
    segment_id: Mapped[str] = mapped_column(ForeignKey("segments.id"), primary_key=True)
    reference: Mapped[str] = mapped_column(Text)
    hypothesis: Mapped[str] = mapped_column(Text)
    wer: Mapped[float] = mapped_column(Float)
    cer: Mapped[float] = mapped_column(Float)
    n_ref_words: Mapped[int] = mapped_column(Integer)
    n_switch_points: Mapped[int] = mapped_column(Integer)
    n_switch_errors: Mapped[int] = mapped_column(Integer)


# ───────────────────────── Jobs & transcription ─────────────────────────
JOB_KINDS = ("parse_transcript", "align", "vad_segment", "hypothesize", "build_dataset", "export_dataset",
             "register_model", "convert_model", "evaluate", "transcribe", "export_transcription")


class Job(Timestamps, Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(Text)
    payload: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(Text, default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    worker_id: Mapped[Optional[str]] = mapped_column(Text)
    heartbeat_at: Mapped[Optional[str]] = mapped_column(Text)
    error: Mapped[Optional[str]] = mapped_column(Text)
    result: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[str]] = mapped_column(Text)
    finished_at: Mapped[Optional[str]] = mapped_column(Text)
    __table_args__ = (CheckConstraint(_in("kind", *JOB_KINDS), name="ck_kind"),
                      CheckConstraint(_in("status", "queued", "running", "done", "failed", "cancelled"),
                                      name="ck_status"),
                      _json("payload"),
                      Index("ix_jobs_queue", "status", "created_at"))


class Transcription(Timestamps, Base):
    __tablename__ = "transcriptions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    title: Mapped[str] = mapped_column(Text)
    input_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(Text)
    duration_s: Mapped[Optional[float]] = mapped_column(Float)
    domain: Mapped[str] = mapped_column(Text, default="other")
    consent_to_train: Mapped[int] = mapped_column(Integer, default=0)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    rtf: Mapped[Optional[float]] = mapped_column(Float)
    expires_at: Mapped[Optional[str]] = mapped_column(Text)
    __table_args__ = (CheckConstraint("consent_to_train IN (0, 1)", name="ck_consent"),)


class TranscriptionSegment(Base):
    __tablename__ = "transcription_segments"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    transcription_id: Mapped[str] = mapped_column(ForeignKey("transcriptions.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)              # raw model output, immutable
    words: Mapped[Optional[str]] = mapped_column(Text)
    avg_logprob: Mapped[Optional[float]] = mapped_column(Float)
    no_speech_prob: Mapped[Optional[float]] = mapped_column(Float)
    edited_text: Mapped[Optional[str]] = mapped_column(Text)
    edited_at: Mapped[Optional[str]] = mapped_column(Text)
    promoted_segment_id: Mapped[Optional[str]] = mapped_column(ForeignKey("segments.id"))
    __table_args__ = (UniqueConstraint("transcription_id", "ordinal"),)
