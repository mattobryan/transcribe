"""Score systems on the corrected chunks: WER, WER by language, and the error rate at the Swahili/English switch points.

    python -m transcribe.train.evaluate data/train/test.jsonl --systems scribe_raw scribe_cleaned hf:openai/whisper-small \
        hf:openai/whisper-small@checkpoints/run1/adapter --out reports/test.json

Systems:
  scribe_raw      Scribe's first text for the chunk (its mm-hmm lines and marks included).
  scribe_cleaned  Scribe after the corrector's mechanical clean-up, before any person edited it.
  hf:<model>[@<adapter dir>]  a Whisper model, optionally with a LoRA adapter, run on the chunk audio.

Scribe's numbers are favourable to Scribe: the reference is Scribe's own text after a person edited it.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from transcribe.pilot.metrics import aggregate, score_segment

from .dataset import read_manifest


def score_system(rows: List[Dict], hypotheses: Dict[str, str]) -> Dict:
    scored = [score_segment(r["ref_words"], r["ref_langs"], hypotheses.get(r["id"], "")) for r in rows]
    result = aggregate(scored, rows)
    # the same interviewer talks in every recording, so also report who is hard: the interviewer (I) or the respondent (R)
    by_speaker = aggregate(scored, [{"ref_langs": r.get("ref_speakers") or ["?"] * len(r["ref_words"])} for r in rows])
    result["wer_by_speaker"] = by_speaker["wer_by_lang"]
    return result


def scribe_hypotheses(rows: List[Dict], kind: str) -> Dict[str, str]:
    return {r["id"]: r[kind] for r in rows}


def transcribe_rows(model, processor, rows: List[Dict], language: Optional[str] = "dominant", batch_size: int = 8, device: str = "cpu") -> Dict[str, str]:
    """Run a Whisper model over the chunk clips. language: 'dominant' (the chunk's main language), 'auto', or a code."""
    import soundfile as sf
    import torch

    model.eval()
    out: Dict[str, str] = {}
    groups: Dict[Optional[str], List[Dict]] = {}
    for r in rows:
        lang = r["language"] if language == "dominant" else (None if language == "auto" else language)
        groups.setdefault(lang, []).append(r)
    for lang, part in groups.items():
        for i in range(0, len(part), batch_size):
            batch = part[i:i + batch_size]
            audio = [sf.read(r["audio"], dtype="float32")[0] for r in batch]
            feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
            kwargs = {"task": "transcribe", "max_new_tokens": 220}
            if lang:
                kwargs["language"] = lang
            with torch.no_grad():
                ids = model.generate(input_features=feats, **kwargs)
            for r, text in zip(batch, processor.batch_decode(ids, skip_special_tokens=True)):
                out[r["id"]] = text.strip()
    return out


def load_hf(spec: str, device: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    name, _, adapter = spec.partition("@")
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.to(device), processor


def report_markdown(results: Dict[str, Dict], split: str, n: int) -> str:
    lines = [f"# Evaluation on {split} ({n} chunks)", "", "WER in percent. sw / en / mixed: by the language of the reference word. I / R: interviewer / respondent.", "", "| system | WER | CER | sw | en | mixed | switch points | I | R |", "|---|---|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        by = m["wer_by_lang"]
        fmt = lambda v: "" if v is None else f"{100 * v:.1f}"          # noqa: E731
        lines.append(f"| {name} | {fmt(m['wer'])} | {fmt(m['cer'])} | {fmt(by.get('sw'))} | {fmt(by.get('en'))} | {fmt(by.get('mixed'))} | {fmt(m['switch_point_error_rate'])} | {fmt(m['wer_by_speaker'].get('I'))} | {fmt(m['wer_by_speaker'].get('R'))} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest")
    parser.add_argument("--systems", nargs="+", default=["scribe_raw", "scribe_cleaned"])
    parser.add_argument("--language", default="dominant", help="dominant (per chunk), auto, or a code such as sw")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = read_manifest(args.manifest)
    results = {}
    for system in args.systems:
        if system in ("scribe_raw", "scribe_cleaned"):
            results[system] = score_system(rows, scribe_hypotheses(rows, system))
        elif system.startswith("hf:"):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, processor = load_hf(system[3:], device)
            results[system] = score_system(rows, transcribe_rows(model, processor, rows, args.language, args.batch, device))
        else:
            raise SystemExit(f"unknown system {system}")
        print(system, "WER", results[system]["wer"], "switch points", results[system]["switch_point_error_rate"], flush=True)
    md = report_markdown(results, Path(args.manifest).stem, len(rows))
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        Path(args.out).with_suffix(".md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
