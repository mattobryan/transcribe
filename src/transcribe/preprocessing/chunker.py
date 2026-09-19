"""Create deterministic WAV chunks from VAD intervals."""

from pathlib import Path
from typing import Dict, List, Optional

import librosa
import soundfile as sf

from .vad import SpeechInterval


def _split_interval(interval: SpeechInterval, max_seconds: float) -> List[SpeechInterval]:
    duration = interval.end - interval.start
    if duration <= max_seconds:
        return [interval]
    chunks = []
    start = interval.start
    while start < interval.end:
        end = min(start + max_seconds, interval.end)
        chunks.append(SpeechInterval(start, end, interval.confidence, interval.backend))
        start = end
    return chunks


def export_chunks(
    audio_path: str,
    intervals: List[SpeechInterval],
    output_dir: str,
    recording_id: str,
    max_seconds: float = 30.0,
    sample_rate: int = 16000,
    transcript_turns: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Write candidate chunks and return manifest segment records.

    Transcript matching is intentionally left for the review/alignment stage.
    """
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_duration = len(audio) / sample_rate
    records = []
    sequence = 0
    for interval in intervals:
        for part in _split_interval(interval, max_seconds):
            start_sample = max(0, int(round(part.start * sample_rate)))
            end_sample = min(len(audio), int(round(part.end * sample_rate)))
            if end_sample <= start_sample:
                continue
            chunk_id = f"{recording_id}_{sequence:06d}"
            file_path = output / f"{chunk_id}.wav"
            sf.write(file_path, audio[start_sample:end_sample], sample_rate, subtype="PCM_16")
            records.append({
                "segment_id": chunk_id,
                "recording_id": recording_id,
                "sequence": sequence,
                "audio_file": str(file_path),
                "source_audio": str(audio_path),
                "start": start_sample / sample_rate,
                "end": end_sample / sample_rate,
                "duration": (end_sample - start_sample) / sample_rate,
                "transcript": "",
                "speaker_id": None,
                "vad": part.to_dict(),
                "status": "candidate",
                "transcript_turn_id": None,
                "source_duration": source_duration,
            })
            sequence += 1
    return records