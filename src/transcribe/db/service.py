"""Service-level invariants (05 Backend Schema, section 3).

1. ``segment_revisions`` is append-only: no ORM update or delete (cascade
   deletes of a whole recording happen in the database).
2. A segment's start/end/status/speaker always equal its current revision's;
   ``add_revision`` is the only way to change them, and it refuses a stale base
   revision so two editors cannot silently overwrite each other (HTTP 409 later).
"""

import json
from typing import Dict, Optional

from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from .models import Segment, SegmentRevision


class RevisionConflict(Exception):
    """The edit was based on a revision that is no longer current."""


class AppendOnlyViolation(Exception):
    """Something tried to change or delete a segment revision."""


@event.listens_for(Session, "before_flush")
def _revisions_are_append_only(session, _context, _instances):
    for obj in list(session.dirty):
        if isinstance(obj, SegmentRevision) and session.is_modified(obj, include_collections=False):
            raise AppendOnlyViolation(f"segment revision {obj.id} cannot be modified")
    for obj in list(session.deleted):
        if isinstance(obj, SegmentRevision):
            raise AppendOnlyViolation(f"segment revision {obj.id} cannot be deleted")


def add_revision(session: Session, segment: Segment, *, text: str, start_s: float, end_s: float,
                 status: str, source: str, author: str, flags: Optional[Dict[str, bool]] = None,
                 speaker_label: Optional[str] = None, base_revision_id: Optional[str] = None,
                 hypothesis_id: Optional[str] = None) -> SegmentRevision:
    """Append a revision and move the segment to it (invariant 2)."""
    if base_revision_id != segment.current_revision_id:
        raise RevisionConflict(
            f"segment {segment.id}: edit based on {base_revision_id}, current is {segment.current_revision_id}")
    number = session.scalar(select(func.coalesce(func.max(SegmentRevision.revision_no), 0))
                            .where(SegmentRevision.segment_id == segment.id)) + 1
    revision = SegmentRevision(
        segment_id=segment.id, revision_no=number, parent_revision_id=segment.current_revision_id,
        text=text, start_s=start_s, end_s=end_s, speaker_label=speaker_label,
        flags=json.dumps(flags or {}), status=status, source=source, author=author,
        hypothesis_id=hypothesis_id)
    session.add(revision)
    session.flush()
    segment.current_revision_id = revision.id
    segment.start_s, segment.end_s = start_s, end_s
    segment.status, segment.speaker_label = status, speaker_label
    return revision
