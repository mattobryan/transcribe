"""Run the pilot on every recording found in a folder and summarise them.

Pairs each transcript (.docx/.pdf) with the audio file whose name matches
(case, spaces and underscores ignored), runs ``pilot.run`` for each pair in its
own process, and writes ``summary.md`` / ``summary.json`` across recordings.

    python -m src.transcribe.pilot.batch --input /kaggle/input --out /kaggle/working/pilot
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4"}
TEXT_EXT = {".docx", ".pdf"}


def key(path: Path) -> str:
    return re.sub(r"[\s_\-.]+", "", path.stem.lower())


def pair_files(root: Path) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Path]]:
    files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith((".", "~$"))]
    audio = {key(p): p for p in files if p.suffix.lower() in AUDIO_EXT}
    texts = {}
    for p in sorted(files):
        if p.suffix.lower() in TEXT_EXT:
            texts.setdefault(key(p), p)       # prefer .docx over .pdf (sorted: d < p)
    pairs = [(audio[k], texts[k]) for k in sorted(texts) if k in audio]
    lonely_text = [texts[k] for k in sorted(texts) if k not in audio]
    lonely_audio = [audio[k] for k in sorted(audio) if k not in texts]
    return pairs, lonely_text, lonely_audio


def summarise(out: Path, results: List[Dict]) -> None:
    rows = []
    for result in results:
        report_path = out / result["recording_id"] / "report.json"
        if not report_path.exists():
            rows.append({**result})
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        a, s = report["alignment"], report["segments"]
        row = {**result, "audio_h": round(report["audio_seconds"] / 3600, 2),
               "anchor_turns": a.get("anchor_turns"), "anchors_below": a.get("turns_below_0_5"),
               "median_conf": (a["turn_confidence_quantiles"] or {}).get("50"),
               "marker_median_err_s": (report.get("marker_check") or {}).get("median_abs_error_s"),
               "marker_within_10s": (report.get("marker_check") or {}).get("within_10s"),
               "segments": s["count"], "clean_h": s["clean_hours"], "asr": {}}
        for entry in report["asr"]:
            row["asr"][f"{entry['model']}/{entry['language']}"] = {
                "wer": entry["clean"]["wer"], "wer_sw": entry["clean"]["wer_by_lang"].get("sw"),
                "wer_en": entry["clean"]["wer_by_lang"].get("en"),
                "switch": entry["clean"]["switch_point_error_rate"],
                "gap_wpm": entry.get("gap_words_per_min"), "rtf": entry.get("rtf")}
        rows.append(row)
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(v):
        return "–" if v is None else f"{100 * v:.1f}%"

    lines = ["# Pilot summary", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}", "",
             "| Recording | Status | Audio h | Anchor turns | Below threshold | Median conf | Marker error (s) | Markers ≤10 s | Segments | Clean h |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['recording_id']} | {r['status']} | {r.get('audio_h', '–')} | {r.get('anchor_turns', '–')} | "
                     f"{r.get('anchors_below', '–')} | {r.get('median_conf', '–')} | {r.get('marker_median_err_s', '–')} | "
                     f"{pct(r.get('marker_within_10s')) if r.get('audio_h') else '–'} | {r.get('segments', '–')} | {r.get('clean_h', '–')} |")
    configs = sorted({c for r in rows for c in r.get("asr", {})})
    if configs:
        lines += ["", "Zero-shot Whisper on clean segments:", "",
                  "| Recording | Config | WER | WER sw | WER en | Switch-point ER | Gap words/min | RTF |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            for c in configs:
                m = r.get("asr", {}).get(c)
                if m:
                    lines.append(f"| {r['recording_id']} | {c} | {pct(m['wer'])} | {pct(m['wer_sw'])} | "
                                 f"{pct(m['wer_en'])} | {pct(m['switch'])} | {m['gap_wpm']} | {m['rtf']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Folder searched recursively")
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="", help="Comma list of recording ids to run")
    args, passthrough = parser.parse_known_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, lonely_text, lonely_audio = pair_files(Path(args.input))
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    print(f"Found {len(pairs)} audio/transcript pairs", flush=True)
    for text in lonely_text:
        print(f"  transcript without audio: {text}", flush=True)
    for audio in lonely_audio:
        print(f"  audio without transcript: {audio}", flush=True)
    results = []
    for audio, text in pairs:
        recording_id = text.stem
        if only and recording_id not in only:
            continue
        print(f"\n=== {recording_id}: {audio.name} + {text.name} ===", flush=True)
        started = time.time()
        command = [sys.executable, "-m", "src.transcribe.pilot.run", "--audio", str(audio),
                   "--transcript", str(text), "--out", str(out / recording_id),
                   "--recording-id", recording_id, *passthrough]
        code = subprocess.call(command)
        results.append({"recording_id": recording_id, "status": "ok" if code == 0 else f"failed ({code})",
                        "minutes": round((time.time() - started) / 60, 1)})
        summarise(out, results)
    summarise(out, results)
    print(f"\nSummary: {out / 'summary.md'}", flush=True)
    if any(r["status"] != "ok" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
