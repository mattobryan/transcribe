"""Pilot on one recording: align transcript, segment, score zero-shot Whisper.

Usage (Windows PowerShell, from the repository root):

    .\\.venv\\Scripts\\python.exe -m src.transcribe.pilot.run `
        --audio "data/audio/NRCCW_ KSM09.MP3" `
        --transcript data/audio/NRCCW_KSM09.docx `
        --out data/pilot/NRCCW_KSM09 --export-review

Every stage is cached in --out, so re-running only redoes what is missing.
Send back report.md and report.json; they contain text and numbers, no audio.
"""

import argparse
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .metrics import aggregate, score_segment
from .segment import Policy, build_segments, silence_gaps
from .textprep import Word, select_turns, turns_to_words

SR = 16000


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def bar(label: str):
    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 20) == 0:
            print(f"    {label}: {done}/{total}", flush=True)
    return progress


def hms(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_transcript(path: str) -> List[Dict]:
    if path.lower().endswith(".pdf"):
        from src.transcribe.preprocessing.pdf_parser import parse_pdf
        manifest = parse_pdf(path)
    else:
        from src.transcribe.preprocessing.docx_parser import parse_docx
        manifest = parse_docx(path)
    return [asdict(turn) for turn in manifest.transcript_turns]


def load_audio(path: str) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio.astype(np.float32)


def align_stage(args, out: Path, audio: np.ndarray, turns: List[Dict]) -> Dict:
    cache = out / "alignment.json"
    if cache.exists() and not args.realign:
        data = read_json(cache)
        data["words"] = [Word(**w) for w in data["words"]]
        return data
    from .ctc_align import Emitter, align_words, load_or_compute_emissions

    log(f"Loading aligner {args.aligner} (first run downloads ~1.2 GB)")
    emitter = Emitter(args.aligner)
    log("Computing emissions (the slow step; cached afterwards)")
    started = time.perf_counter()
    logp = load_or_compute_emissions(emitter, audio, out / "emissions.npy", bar("windows"))
    emission_seconds = time.perf_counter() - started
    words = turns_to_words(turns, alphabet=emitter.alphabet)
    log(f"Aligning {len(turns)} turns / {len(words)} words")
    started = time.perf_counter()
    turn_reports = align_words(logp, words, emitter.token_ids, emitter.blank, progress=bar("turns"))
    data = {"aligner": args.aligner, "emission_seconds": round(emission_seconds, 1),
            "alignment_seconds": round(time.perf_counter() - started, 1),
            "turns": turn_reports, "words": words}
    write_json(cache, {**data, "words": [asdict(w) for w in words]})
    return data


def asr_stage(args, out: Path, audio: np.ndarray, spans: List[Dict], kind: str) -> Dict:
    from .asr import WhisperRunner, run_config

    results = {}
    ids = [s.get("segment_id", f"gap{i}") for i, s in enumerate(spans)]
    for model in args.models:
        runner = None
        for language in args.languages:
            cache = out / f"asr_{model}_{language}_{kind}.json"
            if cache.exists() and not args.rerun_asr:
                cached = read_json(cache)
                if cached.get("ids") == ids:
                    results[(model, language)] = cached
                    continue
            if runner is None:
                log(f"Loading Whisper {model}")
                runner = WhisperRunner(model, threads=args.threads)
            log(f"Whisper {model}, language={language}: {len(spans)} {kind}")
            result = run_config(runner, audio, spans, language, bar(f"{model}/{language}"))
            result["ids"] = ids
            write_json(cache, result)
            results[(model, language)] = result
    return results


def evenly(items: List, limit: int) -> List:
    if not limit or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def export_review(out: Path, audio: np.ndarray, args, turns: List[Dict], segments: List[Dict],
                  hypotheses: Dict[str, str]) -> Path:
    import soundfile as sf

    chunk_dir = out / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for segment in segments:
        path = chunk_dir / f"{segment['segment_id']}.wav"
        if not path.exists():
            sf.write(path, audio[int(segment["start"] * SR):int(segment["end"] * SR)], SR, subtype="PCM_16")
        records.append({
            "segment_id": segment["segment_id"],
            "recording_id": args.recording_id,
            "sequence": segment["sequence"],
            "audio_file": str(path.resolve()),
            "source_audio": args.audio,
            "start": segment["start"], "end": segment["end"], "duration": segment["duration"],
            "transcript": segment["text"],
            "prefill": segment["text"],
            "speaker_id": segment["speaker"],
            "vad": {},
            "status": "candidate",
            "transcript_turn_id": turns[segment["turn_indices"][0]].get("turn_id"),
            "alignment_conf": segment["alignment_conf"],
            "flags": segment["flags"],
            "asr_hypothesis": hypotheses.get(segment["segment_id"], ""),
        })
    manifest = out / "review_manifest.json"
    if manifest.exists():
        log(f"Keeping existing {manifest} (delete it to regenerate; it may hold your review work)")
        return manifest
    write_json(manifest, {
        "schema_version": 1, "recording_id": args.recording_id, "source_audio": args.audio,
        "origin": "pilot_alignment",
        "transcript_turns": [{"turn_id": t.get("turn_id"), "speaker_id": t.get("speaker_id"),
                              "text": t.get("text", "")} for t in turns],
        "segments": records,
    })
    return manifest


def build_report(args, audio_seconds, turns, alignment, segments, eval_segments, clean_ids,
                 gaps, asr, gap_asr) -> Dict:
    words = alignment["words"]
    turn_reports = alignment["turns"]
    status = Counter(t["status"] for t in turn_reports)
    confs = [t["confidence"] for t in turn_reports if t["confidence"] is not None]
    aligned_words = [w for w in words if w.start is not None]
    flag_counts = Counter(flag for s in segments for flag in s["flags"])
    durations = [s["duration"] for s in segments]
    seg_seconds = sum(durations)
    clean_seconds = sum(s["duration"] for s in segments if s["segment_id"] in clean_ids)
    lang_counts = Counter(w.lang for w in words if w.norm)
    low_turns = sorted((t for t in turn_reports if t["confidence"] is not None),
                       key=lambda t: t["confidence"])[:25]

    report = {
        "recording_id": args.recording_id,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "machine": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "processor": platform.processor()},
        "audio_seconds": round(audio_seconds, 1),
        "transcript": {"turns": len(turns), "words": len(words),
                       "speakers": dict(Counter(t.get("speaker_id") for t in turns)),
                       "language_tags": dict(lang_counts)},
        "alignment": {
            "aligner": alignment.get("aligner"),
            "emission_seconds": alignment.get("emission_seconds"),
            "alignment_seconds": alignment.get("alignment_seconds"),
            "turn_status": dict(status),
            "turn_confidence_quantiles": {q: round(float(np.quantile(confs, q / 100)), 3)
                                          for q in (5, 10, 25, 50, 75, 90)} if confs else {},
            "turns_below_0_5": sum(c < 0.5 for c in confs),
            "words_timed": len(aligned_words),
            "lowest_turns": [{
                "turn": t["turn_index"], "turn_id": turns[t["turn_index"]].get("turn_id"),
                "speaker": t["speaker"], "confidence": round(t["confidence"], 3),
                "at": hms(t.get("start") or 0), "text": turns[t["turn_index"]].get("text", "")[:160],
            } for t in low_turns],
        },
        "segments": {
            "count": len(segments), "hours": round(seg_seconds / 3600, 3),
            "clean_count": len(clean_ids), "clean_hours": round(clean_seconds / 3600, 3),
            "duration_quantiles": {q: round(float(np.quantile(durations, q / 100)), 1)
                                   for q in (5, 25, 50, 75, 95)} if durations else {},
            "flags": dict(flag_counts),
            "evaluated": len(eval_segments),
        },
        "asr": [],
        "silence_gaps": {"count": len(gaps), "minutes": round(sum(g["end"] - g["start"] for g in gaps) / 60, 2)},
    }

    by_id = {s["segment_id"]: s for s in eval_segments}
    for (model, language), result in asr.items():
        scored_all, segs_all = [], []
        for seg_id, output in zip(result["ids"], result["outputs"]):
            segment = by_id[seg_id]
            scored = score_segment(segment["ref_words"], segment["ref_langs"], output.get("text", ""))
            scored_all.append(scored)
            segs_all.append(segment)
        clean = [(sc, sg) for sc, sg in zip(scored_all, segs_all) if sg["segment_id"] in clean_ids]
        entry = {
            "model": model, "language": language, "rtf": result.get("rtf"),
            "all": aggregate(scored_all, segs_all),
            "clean": aggregate([c[0] for c in clean], [c[1] for c in clean]),
            "clean_by_speaker": {},
            "detected_languages": dict(Counter(o.get("detected_language") for o in result["outputs"])),
        }
        for speaker in sorted({sg["speaker"] for _, sg in clean}, key=str):
            pairs = [(sc, sg) for sc, sg in clean if sg["speaker"] == speaker]
            entry["clean_by_speaker"][str(speaker)] = aggregate([p[0] for p in pairs], [p[1] for p in pairs])
        worst = sorted(((sc, sg) for sc, sg in clean if sc["n_ref"] >= 5),
                       key=lambda p: p[0]["errors"] / p[0]["n_ref"], reverse=True)[:15]
        entry["worst_clean_segments"] = [{
            "segment_id": sg["segment_id"], "at": hms(sg["start"]), "speaker": sg["speaker"],
            "wer": round(sc["errors"] / sc["n_ref"], 3), "reference": " ".join(sg["ref_words"]),
            "hypothesis": sc["hyp_norm"],
        } for sc, sg in worst]
        gap_result = gap_asr.get((model, language))
        if gap_result:
            gap_words = sum(len(o.get("text", "").split()) for o in gap_result["outputs"])
            minutes = sum(g["end"] - g["start"] for g in gaps) / 60
            entry["gap_words"] = gap_words
            entry["gap_words_per_min"] = round(gap_words / minutes, 2) if minutes else None
            entry["gap_samples"] = [{"at": hms(g["start"]), "text": o.get("text", "")}
                                    for g, o in zip(gaps, gap_result["outputs"]) if o.get("text")][:10]
        report["asr"].append(entry)
    return report


def pct(value) -> str:
    return "–" if value is None else f"{100 * value:.1f}%"


def render_markdown(report: Dict) -> str:
    a, s = report["alignment"], report["segments"]
    lines = [
        f"# Pilot report: {report['recording_id']}",
        "",
        f"Generated {report['generated']} · audio {hms(report['audio_seconds'])} · "
        f"{report['transcript']['turns']} turns · {report['transcript']['words']} words",
        "",
        "## 1. Transcript to audio alignment",
        "",
        f"- Aligner: `{a['aligner']}` · emissions {a['emission_seconds']} s · alignment {a['alignment_seconds']} s",
        f"- Turn status: {a['turn_status']}",
        f"- Turn confidence quantiles: {a['turn_confidence_quantiles']}",
        f"- Turns below 0.5 confidence (likely non-verbatim, missing or misplaced): **{a['turns_below_0_5']}**",
        f"- Language tags from italics: {report['transcript']['language_tags']}",
        f"- Speakers: {report['transcript']['speakers']}",
        "",
        "Lowest-confidence turns (listen to these first):",
        "",
        "| Turn | Speaker | Conf | At | Text |",
        "|---|---|---|---|---|",
    ]
    for t in a["lowest_turns"]:
        text = t["text"].replace("|", "/")
        lines.append(f"| {t['turn_id']} | {t['speaker']} | {t['confidence']} | {t['at']} | {text} |")
    lines += [
        "",
        "## 2. Segments",
        "",
        f"- {s['count']} segments, {s['hours']} h; clean (no flags) {s['clean_count']}, {s['clean_hours']} h",
        f"- Duration quantiles (s): {s['duration_quantiles']}",
        f"- Flags: {s['flags']}",
        f"- Evaluated with Whisper: {s['evaluated']} segments",
        "",
        "## 3. Zero-shot Whisper",
        "",
        "Clean segments only (the fair comparison). WER per language counts substitutions and deletions of reference words in that language.",
        "",
        "| Model | Lang | WER | CER | WER en | WER sw | WER mixed | Switch-point ER | Gap words/min | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in report["asr"]:
        c = e["clean"]
        by = c["wer_by_lang"]
        lines.append(
            f"| {e['model']} | {e['language']} | {pct(c['wer'])} | {pct(c['cer'])} | {pct(by.get('en'))} | "
            f"{pct(by.get('sw'))} | {pct(by.get('mixed'))} | {pct(c['switch_point_error_rate'])} | "
            f"{e.get('gap_words_per_min', '–')} | {e['rtf']} |")
    lines += ["", "All evaluated segments (includes flagged ones):", "",
              "| Model | Lang | WER | CER | Segments | Ref words |", "|---|---|---|---|---|---|"]
    for e in report["asr"]:
        c = e["all"]
        lines.append(f"| {e['model']} | {e['language']} | {pct(c['wer'])} | {pct(c['cer'])} | {c['segments']} | {c['ref_words']} |")
    for e in report["asr"]:
        lines += ["", f"### {e['model']} / {e['language']}", "",
                  f"- Detected languages: {e['detected_languages']}",
                  f"- Clean WER by speaker: " + ", ".join(
                      f"{k}: {pct(v['wer'])} ({v['ref_words']} words)" for k, v in e["clean_by_speaker"].items()),
                  f"- Top substitutions (ref → hyp × count): " + "; ".join(
                      f"{r}→{h}×{n}" for r, h, n in e["clean"]["top_substitutions"][:15])]
        if e.get("gap_samples"):
            lines.append("- Text emitted in untranscribed gaps (hallucination or speech missing from the transcript):")
            lines += [f"  - {g['at']}: {g['text'][:160]}" for g in e["gap_samples"]]
        lines += ["", "| Segment | At | Speaker | WER | Reference | Hypothesis |", "|---|---|---|---|---|---|"]
        for w in e["worst_clean_segments"]:
            lines.append(f"| {w['segment_id']} | {w['at']} | {w['speaker']} | {pct(w['wer'])} | "
                         f"{w['reference'][:140]} | {w['hypothesis'][:140]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-recording alignment + zero-shot Whisper pilot")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript", required=True, help="Reviewed DOCX (or PDF)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--recording-id")
    parser.add_argument("--aligner", default="MahmoudAshraf/mms-300m-1130-forced-aligner")
    parser.add_argument("--models", default="small", help="Comma list, e.g. small,large-v3-turbo")
    parser.add_argument("--languages", default="sw,en,auto")
    parser.add_argument("--max-segments", type=int, default=0, help="Evaluate an even sample (0 = all)")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--export-review", action="store_true", help="Write WAV chunks + review_manifest.json")
    parser.add_argument("--realign", action="store_true")
    parser.add_argument("--rerun-asr", action="store_true")
    args = parser.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    args.recording_id = args.recording_id or Path(args.transcript).stem

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Parsing {args.transcript}")
    turns = select_turns(parse_transcript(args.transcript))
    if not turns:
        sys.exit("No speaker-labelled turns found (expected lines starting with I:, R:, P1: ...)")
    log(f"Loading audio {args.audio}")
    audio = load_audio(args.audio)
    audio_seconds = len(audio) / SR
    log(f"Audio {hms(audio_seconds)}; {len(turns)} turns")

    alignment = align_stage(args, out, audio, turns)
    segments = build_segments(alignment["words"], args.recording_id, audio_seconds, Policy())
    write_json(out / "segments.json", segments)
    eval_segments = [s for s in segments if "too_short" not in s["flags"]
                     and s["alignment_conf"] >= 0.3 and s["ref_words"]]
    eval_segments = evenly(eval_segments, args.max_segments)
    clean_ids = {s["segment_id"] for s in eval_segments if not s["flags"]}
    gaps = silence_gaps(alignment["words"])
    log(f"{len(segments)} segments; evaluating {len(eval_segments)} ({len(clean_ids)} clean); {len(gaps)} silence gaps")

    asr, gap_asr = {}, {}
    if not args.skip_asr:
        asr = asr_stage(args, out, audio, eval_segments, "segments")
        if gaps:
            gap_asr = asr_stage(args, out, audio, gaps, "gaps")

    hypotheses = {}
    if asr:
        first = next(iter(asr.values()))
        hypotheses = {i: o.get("text", "") for i, o in zip(first["ids"], first["outputs"])}
    if args.export_review:
        manifest = export_review(out, audio, args, turns, segments, hypotheses)
        log(f"Review manifest: {manifest}")

    report = build_report(args, audio_seconds, turns, alignment, segments, eval_segments,
                          clean_ids, gaps, asr, gap_asr)
    write_json(out / "report.json", report)
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"Done. Send back {out / 'report.md'} and {out / 'report.json'}")


if __name__ == "__main__":
    main()
