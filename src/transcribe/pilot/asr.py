"""Zero-shot Whisper transcription of segments with faster-whisper.

GPU (float16) when available, CPU int8 otherwise; a GPU whose CUDA/cuDNN
libraries fail to load falls back to CPU instead of stopping the run.
"""

import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import guard

SR = 16000
TEMPERATURES = (0.0, 0.4, 0.8)
RETRY_REPETITION_PENALTY = 1.3
RETRY_NO_REPEAT = 4


def max_tokens(seconds: float) -> int:
    return int(min(440, 24 + 12 * seconds))


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class WhisperRunner:
    def __init__(self, model: str = "small", device: str = "auto", threads: int = 0):
        self.name = model
        self.threads = threads or (os.cpu_count() or 4)
        use_gpu = device == "cuda" or (device == "auto" and cuda_available())
        self.device = "cuda" if use_gpu else "cpu"
        try:
            self.model = self._load()
        except (RuntimeError, ValueError) as exc:         # GPU libraries missing or incompatible
            if self.device != "cuda":
                raise
            print(f"    GPU unavailable ({exc}); using CPU int8", flush=True)
            self.device = "cpu"
            self.model = self._load()

    def _load(self):
        from faster_whisper import WhisperModel

        compute_type = "float16" if self.device == "cuda" else "int8"
        return WhisperModel(self.name, device=self.device, compute_type=compute_type,
                            cpu_threads=self.threads)

    def transcribe(self, audio: np.ndarray, language: Optional[str], initial_prompt: Optional[str] = None,
                   hotwords: Optional[str] = None) -> Dict:
        """``initial_prompt`` and ``hotwords`` bias decoding towards a reviewer's
        recent corrections (vocabulary, spellings) without changing weights.

        A chunk that loops is decoded again with a repetition penalty; whatever is left of a loop
        is collapsed, and the chunk gets a ``flag`` (list of reasons) so a person reviews it."""
        result = self._run(audio, language, initial_prompt, hotwords)
        if guard.has_loop(result["text"]):
            retry = self._run(audio, language, initial_prompt, hotwords,
                              repetition_penalty=RETRY_REPETITION_PENALTY, no_repeat_ngram_size=RETRY_NO_REPEAT)
            if not guard.has_loop(retry["text"]) or len(retry["text"]) < len(result["text"]):
                result = retry
        result["text"], flags = guard.repair(result["text"], result.get("avg_logprob"))
        if flags:
            result["flag"] = flags
        return result

    def _run(self, audio: np.ndarray, language: Optional[str], initial_prompt: Optional[str],
             hotwords: Optional[str], **extra) -> Dict:
        try:
            return self._transcribe(audio, language, initial_prompt, hotwords, **extra)
        except RuntimeError as exc:
            if self.device != "cuda":
                raise
            print(f"    GPU transcription failed ({exc}); falling back to CPU int8", flush=True)
            self.device = "cpu"
            self.model = self._load()
            return self._transcribe(audio, language, initial_prompt, hotwords, **extra)

    def _transcribe(self, audio: np.ndarray, language: Optional[str], initial_prompt: Optional[str] = None,
                    hotwords: Optional[str] = None, **extra) -> Dict:
        segments, info = self.model.transcribe(
            audio.astype(np.float32), language=language, task="transcribe",
            beam_size=5, condition_on_previous_text=False, vad_filter=False,
            without_timestamps=True, initial_prompt=initial_prompt or None, hotwords=hotwords or None,
            # Repetition loops ("kwa kwa kwa ...") otherwise run to 448 tokens and are retried at
            # six temperatures, taking minutes per chunk: cap the length by the audio duration
            # (no speaker exceeds ~12 tokens/s) and retry at three temperatures only.
            max_new_tokens=max_tokens(len(audio) / SR), temperature=TEMPERATURES, **extra,
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
