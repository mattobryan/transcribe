"""Transcription app: upload audio, transcribe in the background, correct chunk by chunk.

Run with ``python -m transcribe.app`` and open http://127.0.0.1:8000.
"""

import json
import queue
import secrets
import shutil
import threading
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import adapt, pipeline
from .store import Store, now, safe_name

STATIC = Path(__file__).parent / "static"
LOOKAHEAD = 3                  # chunks re-transcribed ahead of the reviewer after each save


class Engine:
    """One Whisper model per name, loaded on first use and shared by both workers."""

    def __init__(self, factory: Optional[Callable[[str], object]] = None):
        self.factory = factory or self._whisper
        self.models: Dict[str, object] = {}
        self.lock = threading.Lock()

    @staticmethod
    def _whisper(name: str):
        from transcribe.pilot.asr import WhisperRunner
        return WhisperRunner(name)

    def transcribe(self, model: str, audio, language, initial_prompt=None, hotwords=None) -> Dict:
        with self.lock:
            if model not in self.models:
                self.models[model] = self.factory(model)
            runner = self.models[model]
            if initial_prompt or hotwords:
                return runner.transcribe(audio, language, initial_prompt=initial_prompt, hotwords=hotwords)
            return runner.transcribe(audio, language)


class ChunkEdit(BaseModel):
    text: str


def render_export(session: Dict, format: str):
    """Transcript as (text, media type): txt (grouped by speaker), srt or json."""
    segs = session["segments"]
    if format == "json":
        return json.dumps(session, ensure_ascii=False, indent=1), "application/json"
    if format == "srt":
        def ts(x: float) -> str:
            ms = int(round(x * 1000))
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        return "\n".join(f"{i + 1}\n{ts(s['start'])} --> {ts(s['end'])}\n"
                         f"{(s['speaker_id'] + ': ') if s.get('speaker_id') else ''}{s['transcript']}\n"
                         for i, s in enumerate(segs)), "application/x-subrip"
    if format == "txt":
        lines, last = [], None
        for s in segs:
            speaker = s.get("speaker_id")
            if speaker and speaker != last:
                lines.append(f"\n{speaker}:")
                last = speaker
            lines.append(s["transcript"])
        return "\n".join(lines).strip() + "\n", "text/plain"
    raise ValueError("format must be txt, srt or json")


def create_app(data_dir: Path = Path("data/app"), transcriber_factory=None, segmenter=None,
               token: Optional[str] = None) -> FastAPI:
    store = Store(Path(data_dir) / "sessions")
    engine = Engine(transcriber_factory)
    jobs: "queue.Queue[str]" = queue.Queue()
    ahead: "queue.Queue[tuple]" = queue.Queue()
    app = FastAPI(title="Transcription")

    if token:
        # For a public link (Colab / Kaggle tunnel): open it once with ?token=..., the browser
        # then keeps a cookie. Every other request without the token is refused.
        @app.middleware("http")
        async def require_token(request, call_next):
            given = request.query_params.get("token") or request.cookies.get("transcribe_token")
            if not secrets.compare_digest(given or "", token):
                return JSONResponse({"detail": "Open the link that includes ?token=..."}, status_code=401)
            response = await call_next(request)
            if request.query_params.get("token"):
                response.set_cookie("transcribe_token", token, httponly=True, samesite="lax", max_age=86400)
            return response
    app.state.store, app.state.engine, app.state.jobs, app.state.ahead = store, engine, jobs, ahead

    # ------------------------------------------------------------ workers
    def run_transcription(session_id: str) -> None:
        folder = store.folder(session_id)
        session = store.load(session_id)
        model = session["settings"].get("model", "small")

        class Bound:                              # binds the session's model to the shared engine
            def transcribe(self, audio, language):
                return engine.transcribe(model, audio, language)

        try:
            kwargs = {"segmenter": segmenter} if segmenter else {}
            pipeline.transcribe_session(session, folder, Bound(), store.save, **kwargs)
        except Exception as exc:
            session.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                           traceback=traceback.format_exc()[-4000:])
            store.save(session)

    def run_lookahead(session_id: str, indices: List[int]) -> None:
        session = store.load(session_id)
        model = session["settings"].get("model", "small")
        language = session["settings"].get("language")
        language = None if language in (None, "auto") else language
        learned = session.get("adaptation", {})
        wav = store.folder(session_id) / "audio16k.wav"
        for index in indices:
            seg = session["segments"][index]
            if seg.get("edited"):
                continue
            clip = pipeline.read_wav(wav, seg["start"], seg["end"])
            result = engine.transcribe(model, clip, language, initial_prompt=learned.get("prompt"),
                                       hotwords=" ".join(learned.get("hotwords", [])) or None)

            def apply(s: Dict, index=index, text=result.get("text", "")) -> None:
                target = s["segments"][index]
                if target.get("edited"):            # the reviewer got there first: never overwrite
                    return
                rules = s.get("adaptation", {}).get("rules", [])
                target["asr_hypothesis"] = text
                target["transcript"] = adapt.apply_rules(text, rules)
                target["suggestion"] = "lookahead"
                target["updated_at"] = now()
            store.update(session_id, apply)

    def worker(q, handler) -> None:
        while True:
            item = q.get()
            try:
                handler(*item) if isinstance(item, tuple) else handler(item)
            except Exception:
                traceback.print_exc()
            finally:
                q.task_done()

    threading.Thread(target=worker, args=(jobs, run_transcription), daemon=True).start()
    threading.Thread(target=worker, args=(ahead, run_lookahead), daemon=True).start()

    # ------------------------------------------------------------ helpers
    def get_session(session_id: str) -> Dict:
        if not store.exists(session_id):
            raise HTTPException(404, "No such transcription")
        return store.load(session_id)

    # ------------------------------------------------------------ routes
    @app.get("/api/sessions")
    def list_sessions():
        return store.list()

    @app.post("/api/sessions")
    async def upload(file: UploadFile = File(...), language: str = Form("sw"), model: str = Form("small"),
                     speakers: bool = Form(True)):
        name = safe_name(file.filename or "audio")
        session = store.create(title=Path(name).stem, source_audio="", settings={
            "language": language, "model": model, "speakers": speakers, "filename": name})
        folder = store.folder(session["id"])
        target = folder / f"original{Path(name).suffix.lower() or '.bin'}"
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        session["source_audio"] = str(target)
        store.save(session)
        jobs.put(session["id"])
        return {"id": session["id"], "position": jobs.qsize()}

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str):
        session = get_session(session_id)
        session.pop("traceback", None)
        return session

    @app.get("/api/sessions/{session_id}/chunks/{index}/audio")
    def chunk_audio(session_id: str, index: int):
        import soundfile as sf
        session = get_session(session_id)
        if not 0 <= index < len(session["segments"]):
            raise HTTPException(404, "No such chunk")
        seg = session["segments"][index]
        # Served as a file so the browser can seek (F1/F2): it needs HTTP range requests.
        cache = store.folder(session_id) / "chunks"
        cache.mkdir(exist_ok=True)
        path = cache / f"{index:05d}_{int(seg['start'] * 1000)}_{int(seg['end'] * 1000)}.wav"
        if not path.exists():
            clip = pipeline.read_wav(store.folder(session_id) / "audio16k.wav", seg["start"], seg["end"])
            tmp = path.with_suffix(".tmp")
            sf.write(tmp, clip, pipeline.SR, format="WAV", subtype="PCM_16")
            tmp.replace(path)
        return FileResponse(path, media_type="audio/wav")

    @app.put("/api/sessions/{session_id}/chunks/{index}")
    def save_chunk(session_id: str, index: int, edit: ChunkEdit):
        get_session(session_id)
        changed: List[int] = []

        def apply(s: Dict) -> None:
            if s.get("status") != "ready":
                raise HTTPException(409, "Transcription is not finished yet")
            if not 0 <= index < len(s["segments"]):
                raise HTTPException(404, "No such chunk")
            seg = s["segments"][index]
            seg.update(transcript=edit.text.strip(), edited=True, status="approved", reviewed_at=now())
            learned = adapt.learn(s["segments"])
            s["adaptation"] = learned
            for i, other in enumerate(s["segments"]):          # apply rules to every chunk not yet edited
                if other.get("edited"):
                    continue
                text = adapt.apply_rules(other.get("asr_hypothesis", ""), learned["rules"])
                if text != other.get("transcript"):
                    other["transcript"] = text
                    other["suggestion"] = "adapted"
                    other["updated_at"] = now()
                    changed.append(i)

        session = store.update(session_id, apply)
        upcoming = [i for i in range(index + 1, len(session["segments"]))
                    if not session["segments"][i].get("edited")][:LOOKAHEAD]
        if upcoming and (session["adaptation"]["prompt"] or session["adaptation"]["hotwords"]):
            ahead.put((session_id, upcoming))
        return {"ok": True, "adapted": changed, "lookahead": upcoming, "adaptation": session["adaptation"]}

    @app.get("/api/sessions/{session_id}/export")
    def export(session_id: str, format: str = "txt"):
        session = get_session(session_id)
        title = safe_name(session.get("recording_id") or "transcript")
        try:
            body, media = render_export(session, format)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return Response(body, media_type=f"{media}; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{title}.{format}"'})

    @app.exception_handler(KeyError)
    def bad_id(_request, _exc):
        return JSONResponse({"detail": "No such transcription"}, status_code=404)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def import_manifest(store: Store, manifest_path: Path) -> str:
    """Open a pilot review manifest (``review_manifest.json``) as an editable session."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    session = store.create(title=manifest.get("recording_id") or Path(manifest_path).parent.name,
                           source_audio=manifest["source_audio"],
                           settings={"language": "sw", "model": "small", "speakers": False})
    folder = store.folder(session["id"])
    duration = pipeline.decode_to_wav(Path(manifest["source_audio"]), folder / "audio16k.wav")
    session["segments"] = [{
        "segment_id": s.get("segment_id", f"{session['id']}_{i:05d}"), "sequence": i,
        "start": s["start"], "end": s["end"], "duration": round(s["end"] - s["start"], 3),
        "transcript": s.get("transcript", ""), "asr_hypothesis": s.get("asr_hypothesis") or s.get("transcript", ""),
        "suggestion": "transcript", "status": "candidate", "edited": False, "speaker_id": s.get("speaker_id"),
        "flags": s.get("flags", []),
    } for i, s in enumerate(manifest.get("segments", []))]
    session.update(status="ready", duration=round(duration, 2),
                   progress={"stage": "Done", "done": len(session["segments"]), "total": len(session["segments"])})
    store.save(session)
    return session["id"]
