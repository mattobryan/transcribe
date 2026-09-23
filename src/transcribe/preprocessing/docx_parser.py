"""Parse reviewed interview DOCX files without changing their wording."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from docx import Document

from .schema import Manifest, TextSpan, TranscriptTurn


SPEAKER_PATTERN = re.compile(
    r"^\s*(?P<label>I|R|P\d+|INTERVIEWER|RESPONDENT\s*\d*)\s*:\s*(?P<text>.*)$",
    re.IGNORECASE,
)
ANNOTATION_PATTERN = re.compile(r"\[[^\]]+\]")


def _speaker_info(label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not label:
        return None, None
    normalized = re.sub(r"\s+", "", label).upper()
    if normalized in {"I", "INTERVIEWER"}:
        return "interviewer", "interviewer"
    if normalized == "R" or normalized.startswith("RESPONDENT"):
        suffix = normalized.replace("RESPONDENT", "").strip() if normalized != "R" else None
        suffix = suffix or None
        return f"respondent_{suffix}" if suffix else "respondent", "respondent"
    if normalized.startswith("P"):
        return normalized, "respondent"
    return normalized, None


def _parse_paragraph(paragraph, paragraph_index: int) -> Optional[TranscriptTurn]:
    source_text = paragraph.text.strip()
    if not source_text:
        return None

    match = SPEAKER_PATTERN.match(source_text)
    label = match.group("label") if match else None
    text_start = match.start("text") if match else 0
    text = match.group("text").strip() if match else source_text
    speaker_id, speaker_role = _speaker_info(label)

    spans: List[TextSpan] = []
    cursor = 0
    for run in paragraph.runs:
        run_text = run.text
        if not run_text:
            continue
        run_start = cursor - text_start
        run_end = run_start + len(run_text)
        cursor += len(run_text)
        if run_end <= 0 or run_start >= len(text):
            continue
        clipped_start = max(0, run_start)
        clipped_end = min(len(text), run_end)
        value = text[clipped_start:clipped_end]
        if not value:
            continue
        spans.append(TextSpan(
            text=value,
            language_hint="sw" if run.italic else "en",
            italic=bool(run.italic),
            start_char=clipped_start,
            end_char=clipped_end,
        ))

    if not spans:
        spans = [TextSpan(text=text, language_hint="unknown", italic=False,
                          start_char=0, end_char=len(text))]

    annotations = ANNOTATION_PATTERN.findall(text)
    return TranscriptTurn(
        turn_id=f"p{paragraph_index:04d}",
        paragraph_index=paragraph_index,
        speaker_id=speaker_id,
        speaker_role=speaker_role,
        text=text,
        spans=spans,
        annotations=annotations,
        source_text=source_text,
    )


def parse_docx(
    path: str,
    recording_id: Optional[str] = None,
    source_audio: Optional[str] = None,
) -> Manifest:
    """Parse a reviewed DOCX into ordered transcript turns and metadata."""
    docx_path = Path(path)
    document = Document(str(docx_path))
    turns = []
    for index, paragraph in enumerate(document.paragraphs):
        turn = _parse_paragraph(paragraph, index)
        if turn is not None:
            turns.append(turn)
    return Manifest(
        recording_id=recording_id or docx_path.stem,
        source_audio=source_audio,
        source_docx=str(docx_path),
        transcript_turns=turns,
        metadata={"parser": "python-docx", "source_format": "docx"},
    )