"""Offline energy-based voice activity detection."""

from dataclasses import asdict, dataclass
from typing import List

import librosa
import numpy as np


@dataclass
class SpeechInterval:
    start: float
    end: float
    confidence: float
    backend: str = "energy"

    def to_dict(self):
        return asdict(self)


def _merge_intervals(intervals: List[SpeechInterval], max_gap: float) -> List[SpeechInterval]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        if current.start - previous.end <= max_gap:
            previous.end = current.end
            previous.confidence = min(previous.confidence, current.confidence)
        else:
            merged.append(current)
    return merged


def detect_speech(
    audio_path: str,
    sample_rate: int = 16000,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold_db: float = -35.0,
    min_speech_seconds: float = 0.25,
    min_silence_seconds: float = 0.6,
    padding_seconds: float = 0.12,
) -> List[SpeechInterval]:
    """Return candidate speech intervals; boundaries are suggestions for review."""
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    if audio.size == 0:
        return []

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    )[0]
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-10), ref=np.max)
    active = db >= threshold_db

    intervals: List[SpeechInterval] = []
    start_frame = None
    for frame_index, is_active in enumerate(np.append(active, False)):
        if is_active and start_frame is None:
            start_frame = frame_index
        elif not is_active and start_frame is not None:
            start = max(0.0, librosa.frames_to_time(start_frame, sr=sample_rate, hop_length=hop_length) - padding_seconds)
            end_frame = frame_index
            end = min(len(audio) / sample_rate, librosa.frames_to_time(end_frame, sr=sample_rate, hop_length=hop_length) + padding_seconds)
            if end - start >= min_speech_seconds:
                frame_db = db[start_frame:frame_index]
                confidence = float(np.clip((np.mean(frame_db) - threshold_db) / max(-threshold_db, 1.0), 0.0, 1.0))
                intervals.append(SpeechInterval(start, end, confidence))
            start_frame = None

    return _merge_intervals(intervals, max_gap=min_silence_seconds)