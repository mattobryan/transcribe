"""Parse text-based interview PDFs into ordered transcript turns."""

import re
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader

from .docx_parser import ANNOTATION_PATTERN, SPEAKER_PATTERN, _speaker_info
from .schema import Manifest, TextSpan, TranscriptTurn


PAGE_HEADER = re.compile(r"^(?:NRCCW[_\s].*|Page\s+\d+\s+of\s+\d+)$", re.IGNORECASE)


def parse_pdf(
    path: str,
    recording_id: Optional[str] = None,
    source_audio: Optional[str] = None,
) -> Manifest:
    """Parse a text-extracted PDF while preserving speaker turn order."""
    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    raw_turns = []
    current_label = None
    current_lines: List[str] = []
    paragraph_index = 0

    def flush() -> None:
        nonlocal paragraph_index, current_lines
        text = " ".join(line.strip() for line in current_lines if line.strip()).strip()
        if not text:
            current_lines = []
            return
        speaker_id, speaker_role = _speaker_info(current_label)
        raw_turns.append(TranscriptTurn(
            turn_id=f"p{paragraph_index:04d}",
            paragraph_index=paragraph_index,
            speaker_id=speaker_id,
            speaker_role=speaker_role,
            text=text,
            spans=[TextSpan(text=text, language_hint="unknown", italic=False,
                            start_char=0, end_char=len(text))],
            annotations=ANNOTATION_PATTERN.findall(text),
            source_text=text,
        ))
        paragraph_index += 1
        current_lines = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            line = " ".join(raw_line.split())
            if not line or PAGE_HEADER.match(line):
                continue
            match = SPEAKER_PATTERN.match(line)
            if match:
                flush()
                current_label = match.group("label")
                current_lines = [match.group("text").strip()]
            elif current_label is not None:
                current_lines.append(line)
    flush()

    return Manifest(
        recording_id=recording_id or pdf_path.stem,
        source_audio=source_audio,
        source_docx=str(pdf_path),
        transcript_turns=raw_turns,
        metadata={
            "parser": "pypdf",
            "source_format": "pdf",
            "language_hints": "not preserved by plain PDF extraction",
        },
    )