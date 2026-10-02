"""Audio in, chunks with model text out.

decode (any format, via PyAV in faster-whisper) -> 16 kHz mono WAV in the session
folder -> Silero VAD speech regions -> chunks of up to 30 s split at pauses ->
Whisper per chunk -> optional speaker labels (pyannote, if installed and a
Hugging Face token is set).
"""

import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

SR = 16000
MAX_CHUNK_S = 30.0
TARGET_CHUNK_S = 15.0
SPLIT_GAP_S = 0.6


def _decode_pyav(source: Path) -> np.ndarray:
    """Audio track of any audio/video file as 16 kHz mono float32, using PyAV directly
    (faster-whisper's own decoder breaks with some PyAV versions)."""
    import av
    chunks = []
    with av.open(str(source)) as container:
        if not container.streams.audio:
            raise ValueError(f"{source.name} has no audio track")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"{source.name}: no audio could be decoded")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _decode_ffmpeg(source: Path) -> np.ndarray:
    import subprocess
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg")
    raw = subprocess.run([exe, "-nostdin", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def _decode_librosa(source: Path) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(str(source), sr=SR, mono=True)
    return audio.astype(np.float32)


def decode_to_wav(source: Path, target: Path) -> float:
    """Decode any audio/video file to 16 kHz mono 16-bit WAV; returns duration in seconds.

    Tries PyAV, then the ffmpeg program, then librosa, and reports every failure if all fail."""
    import soundfile as sf
    errors = []
    for label, decoder in (("PyAV", _decode_pyav), ("ffmpeg", _decode_ffmpeg), ("librosa", _decode_librosa)):
        try:
            audio = decoder(source)
            if len(audio):
                sf.write(target, audio, SR, subtype="PCM_16")
                return len(audio) / SR
            errors.append(f"{label}: empty result")
        except Exception as exc:                                  # try the next decoder
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not read the audio of {source.name}:\n  " + "\n  ".join(errors))


def read_wav(path: Path, start: float = 0.0, end: Optional[float] = None) -> np.ndarray:
    import soundfile as sf
    with sf.SoundFile(path) as f:
        first = max(0, int(start * SR))
        last = f.frames if end is None else min(f.frames, int(end * SR))
        f.seek(first)
        return f.read(max(0, last - first), dtype="float32")


def speech_regions(audio: np.ndarray) -> List[Dict]:
    """Silero VAD speech regions in seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    options = VadOptions(min_silence_duration_ms=400, speech_pad_ms=150, max_speech_duration_s=MAX_CHUNK_S)
    return [{"start": r["start"] / SR, "end": r["end"] / SR} for r in get_speech_timestamps(audio, options)]


def chunk_regions(regions: List[Dict], duration: float) -> List[Dict]:
    """Merge speech regions into chunks: close a chunk at a pause once it is long
    enough, never exceed MAX_CHUNK_S; a single long region is cut evenly."""
    chunks: List[Dict] = []
    current: Optional[Dict] = None
    for region in regions:
        start, end = region["start"], region["end"]
        while end - start > MAX_CHUNK_S:                    # very long region: cut it
            if current:
                chunks.append(current)
                current = None
            chunks.append({"start": start, "end": start + MAX_CHUNK_S})
            start += MAX_CHUNK_S
        if current is None:
            current = {"start": start, "end": end}
            continue
        gap = start - current["end"]
        length = current["end"] - current["start"]
        if end - current["start"] > MAX_CHUNK_S or (gap >= SPLIT_GAP_S and length >= TARGET_CHUNK_S) or gap > 3.0:
            chunks.append(current)
            current = {"start": start, "end": end}
        else:
            current["end"] = end
    if current:
        chunks.append(current)
    return [{"start": round(max(0.0, c["start"]), 3), "end": round(min(duration, c["end"]), 3)} for c in chunks
            if c["end"] - c["start"] >= 0.3]


def diarize(wav: Path, chunks: List[Dict]) -> Optional[List[Optional[str]]]:
    """Speaker label per chunk ("Speaker 1", ...) with pyannote, or None if unavailable."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return None
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return None
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    annotation = pipeline(str(wav))
    names: Dict[str, str] = {}
    labels = []
    for chunk in chunks:
        overlap: Dict[str, float] = {}
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            shared = min(turn.end, chunk["end"]) - max(turn.start, chunk["start"])
            if shared > 0:
                overlap[speaker] = overlap.get(speaker, 0.0) + shared
        if overlap:
            speaker = max(overlap, key=overlap.get)
            names.setdefault(speaker, f"Speaker {len(names) + 1}")
            labels.append(names[speaker])
        else:
            labels.append(None)
    return labels


def transcribe_session(session: Dict, folder: Path, transcriber, save: Callable[[Dict], None],
                       segmenter: Callable[[np.ndarray], List[Dict]] = speech_regions) -> Dict:
    """Run the whole pipeline for a session, saving progress as it goes."""
    settings = session.get("settings", {})
    language = settings.get("language") or None
    if language == "auto":
        language = None
    wav = folder / "audio16k.wav"

    session.update(status="transcribing", progress={"stage": "Preparing the audio", "done": 0, "total": 0})
    save(session)
    duration = decode_to_wav(Path(session["source_audio"]), wav)
    session["duration"] = round(duration, 2)

    session["progress"] = {"stage": "Finding speech", "done": 0, "total": 0}
    save(session)
    audio = read_wav(wav)
    chunks = chunk_regions(segmenter(audio), duration)
    session["segments"] = [{
        "segment_id": f"{session['id']}_{i:05d}", "sequence": i, "start": c["start"], "end": c["end"],
        "duration": round(c["end"] - c["start"], 3), "transcript": "", "asr_hypothesis": "",
        "suggestion": "model", "status": "candidate", "edited": False, "speaker_id": None,
    } for i, c in enumerate(chunks)]

    total = len(chunks)
    for i, seg in enumerate(session["segments"]):
        session["progress"] = {"stage": "Transcribing", "done": i, "total": total}
        save(session)
        clip = audio[int(seg["start"] * SR):int(seg["end"] * SR)]
        result = transcriber.transcribe(clip, language)
        seg["asr_hypothesis"] = seg["transcript"] = result.get("text", "")
        seg["detected_language"] = result.get("detected_language")
        if result.get("flag"):
            seg["flag"] = result["flag"]

    session["progress"] = {"stage": "Identifying speakers", "done": total, "total": total}
    save(session)
    try:
        labels = diarize(wav, chunks) if settings.get("speakers", True) else None
    except Exception as exc:                                    # speaker labels are optional
        labels = None
        session["speaker_error"] = str(exc)
    if labels:
        session["speakers"] = True
        for seg, label in zip(session["segments"], labels):
            seg["speaker_id"] = label

    session.update(status="ready", progress={"stage": "Done", "done": total, "total": total})
    save(session)
    return session
