"""Zero-shot Whisper transcription of segments with faster-whisper (CPU int8)."""

import os
import time
from typing import Dict, List, Optional

import numpy as np

SR = 16000


class WhisperRunner:
    def __init__(self, model: str = "small", compute_type: str = "int8", threads: int = 0):
        from faster_whisper import WhisperModel

        self.name = model
        self.model = WhisperModel(model, device="cpu", compute_type=compute_type,
                                  cpu_threads=threads or (os.cpu_count() or 4))

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> Dict:
        segments, info = self.model.transcribe(
            audio.astype(np.float32), language=language, task="transcribe",
            beam_size=5, condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True,
        )
        parts = list(segments)
        text = " ".join(part.text.strip() for part in parts).strip()
        return {
            "text": text,
            "avg_logprob": float(np.mean([p.avg_logprob for p in parts])) if parts else None,
            "no_speech_prob": float(np.max([p.no_speech_prob for p in parts])) if parts else None,
            "detected_language": info.language,
            "language_probability": round(float(info.language_probability), 3),
        }


def run_config(runner: WhisperRunner, audio: np.ndarray, spans: List[Dict], language: str,
               progress=None) -> Dict:
    """Transcribe each span ({start, end, ...}); returns outputs and real-time factor."""
    lang = None if language == "auto" else language
    outputs = []
    started = time.perf_counter()
    audio_seconds = 0.0
    for i, span in enumerate(spans):
        clip = audio[int(span["start"] * SR):int(span["end"] * SR)]
        audio_seconds += len(clip) / SR
        outputs.append(runner.transcribe(clip, lang) if len(clip) else {"text": ""})
        if progress:
            progress(i + 1, len(spans))
    elapsed = time.perf_counter() - started
    return {"outputs": outputs, "seconds": round(elapsed, 1),
            "audio_seconds": round(audio_seconds, 1),
            "rtf": round(elapsed / audio_seconds, 3) if audio_seconds else None}
