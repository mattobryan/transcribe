"""Transcription sessions on disk: one folder per session.

``session.json`` uses the same shape as the pilot's review manifest
(``recording_id``, ``source_audio``, ``segments`` with ``start``/``end``/
``transcript``/``status``/``reviewed_at``), so ``transcribe.db.migrate_json``
and ``transcribe.pilot.review_stats`` work on it unchanged.
"""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()


def safe_name(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", Path(name).name).strip() or "audio"


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, session_id: str) -> Path:
        if not re.fullmatch(r"[\w\-]+", session_id):
            raise KeyError(session_id)
        return self.root / session_id

    def path(self, session_id: str) -> Path:
        return self.folder(session_id) / "session.json"

    def exists(self, session_id: str) -> bool:
        try:
            return self.path(session_id).exists()
        except KeyError:
            return False

    def load(self, session_id: str) -> Dict:
        with _LOCK:
            return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def save(self, session: Dict) -> None:
        with _LOCK:
            path = self.path(session["id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, path)

    def update(self, session_id: str, fn) -> Dict:
        """Load, apply ``fn(session)`` and save atomically under the store lock."""
        with _LOCK:
            session = self.load(session_id)
            fn(session)
            self.save(session)
            return session

    def create(self, title: str, source_audio: str, settings: Dict, session_id: Optional[str] = None) -> Dict:
        session = {
            "schema_version": 1, "id": session_id or new_session_id(), "recording_id": title,
            "source_audio": source_audio, "origin": "app", "created_at": now(),
            "status": "queued", "progress": {"stage": "queued", "done": 0, "total": 0}, "error": None,
            "settings": settings, "duration": None, "speakers": False,
            "adaptation": {"rules": [], "prompt": "", "hotwords": []},
            "segments": [],
        }
        self.save(session)
        return session

    def list(self) -> List[Dict]:
        out = []
        for path in sorted(self.root.glob("*/session.json"), reverse=True):
            try:
                s = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            segs = s.get("segments", [])
            out.append({"id": s["id"], "title": s.get("recording_id"), "status": s.get("status"),
                        "created_at": s.get("created_at"), "duration": s.get("duration"),
                        "chunks": len(segs), "edited": sum(1 for x in segs if x.get("edited"))})
        return out
