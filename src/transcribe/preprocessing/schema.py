"""Versioned records used while preparing speech training data."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TextSpan:
    text: str
    language_hint: str
    italic: bool
    start_char: int
    end_char: int


@dataclass
class TranscriptTurn:
    turn_id: str
    paragraph_index: int
    speaker_id: Optional[str]
    speaker_role: Optional[str]
    text: str
    spans: List[TextSpan] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    source_text: str = ""
    status: str = "candidate"


@dataclass
class Manifest:
    schema_version: int = 1
    recording_id: str = ""
    source_audio: Optional[str] = None
    source_docx: Optional[str] = None
    transcript_turns: List[TranscriptTurn] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)