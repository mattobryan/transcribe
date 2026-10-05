"""Turn the corrector's exports into a training and evaluation set.

Inputs, per recording (all made by the Transcript Corrector page):
  * the corrections log (``... corrections log.json``): every chunk with Scribe's first text, the page's clean-up and the
    final human text, and whether the chunk was checked;
  * the language labels (``... language labels.csv``): every word of the final text with sw / en / name / mixed;
  * the recording itself (audio or video).

Only chunks a person marked as checked are used. A recording is never split between train, dev and test.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transcribe.pilot.textprep import normalize_word

LABEL = re.compile(r"^[ \t]*[IR]\d*:[ \t]*", re.M)
SPLITS = ("train", "dev", "test")
MAX_SECONDS = 30.0            # Whisper's window
MIN_SECONDS = 0.4


def strip_marks(text: str) -> str:
    """The corrector's marks: **bold**, _Swahili_, ~not Swahili~."""
    return re.sub(r"~(.+?)~", r"\1", re.sub(r"_(.+?)_", r"\1", re.sub(r"\*\*(.+?)\*\*", r"\1", text)))


def training_text(final: str) -> str:
    """What a model should write for the chunk: the words, in order, with the project's dash and ellipsis marks dropped."""
    text = strip_marks(LABEL.sub("", final))
    text = re.sub(r"\.{2,}|…", " ", text)
    text = re.sub(r"(?<=\w)-+(?=\s|$)", " ", text)           # a cut-off or self-correction mark at the end of a word
    text = re.sub(r"(?:(?<=\s)|^)-+(?=\w)", " ", text)       # ... or at the start of one; hyphens inside words stay
    return re.sub(r"\s+", " ", text).strip()


def read_log(path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_labels(path: Optional[str]) -> Dict[int, List[Dict]]:
    """chunk number -> its labelled words, in order."""
    by_chunk: Dict[int, List[Dict]] = {}
    if not path:
        return by_chunk
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_chunk.setdefault(int(row["chunk"]), []).append(row)
    return by_chunk


def reference(rows: List[Dict], fallback_text: str = "") -> Tuple[List[str], List[str], List[str]]:
    """Words for scoring (same normalisation as the metrics), the language of each, and who said each (I or R)."""
    words, langs, speakers = [], [], []
    for row in rows:
        word = normalize_word(row["token"])
        if word:
            words.append(word)
            langs.append(row["language"] if row["language"] in ("sw", "en", "mixed", "name") else "unknown")
            speakers.append(row.get("speaker") or "?")
    if not words and fallback_text:
        words = [w for w in (normalize_word(t) for t in fallback_text.split()) if w]
        langs = ["unknown"] * len(words)
        speakers = ["?"] * len(words)
    return words, langs, speakers


def dominant_language(langs: List[str]) -> str:
    """The language token Whisper is given for the chunk: Swahili when it has more Swahili than English words."""
    sw, en = langs.count("sw"), langs.count("en")
    return "sw" if sw > en else "en"


def slug(name: str) -> str:
    return re.sub(r"[^\w]+", "_", name).strip("_").lower() or "recording"


def build_rows(rec: Dict, wav, clips_dir: Path, only_checked: bool = True, base: Optional[Path] = None) -> List[Dict]:
    """Cut one recording into chunk clips and describe them. ``wav`` is the recording as a 16 kHz mono WAV.
    Clip paths are stored relative to ``base`` (the manifest folder) so the folder can be moved or uploaded."""
    import soundfile as sf

    log, labels = read_log(rec["log"]), read_labels(rec.get("labels"))
    name = slug(rec["name"])
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with sf.SoundFile(wav) as audio:
        rate = audio.samplerate
        for chunk in log["chunks"]:
            if only_checked and not chunk.get("checked"):
                continue
            duration = chunk["end"] - chunk["start"]
            text = training_text(chunk["final"])
            if not text or duration < MIN_SECONDS or duration > MAX_SECONDS + 0.5:
                continue
            ref_words, ref_langs, ref_speakers = reference(labels.get(chunk["chunk"], []), text)
            audio.seek(int(chunk["start"] * rate))
            samples = audio.read(int(duration * rate), dtype="float32")
            path = clips_dir / f"{name}_{chunk['chunk']:05d}.wav"
            sf.write(path, samples, rate, subtype="PCM_16")
            rows.append({
                "id": f"{name}_{chunk['chunk']:05d}", "recording": rec["name"], "split": rec.get("split", "train"),
                "chunk": chunk["chunk"], "audio": str(path.relative_to(base)) if base else str(path), "start": chunk["start"], "end": chunk["end"], "duration": round(duration, 2),
                "text": text, "ref_words": ref_words, "ref_langs": ref_langs, "ref_speakers": ref_speakers, "language": dominant_language(ref_langs),
                "changed": bool(chunk.get("changed")),
                "scribe_raw": " ".join(r["text"] for r in chunk.get("scribe_raw", [])),
                "scribe_cleaned": training_text(chunk.get("after_automatic_clean_up", "")),
            })
    return rows


def write_manifests(rows: List[Dict], out: Path) -> Dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
        counts[split] = len(part)
    return counts


def read_manifest(path) -> List[Dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:                       # clip paths are relative to the manifest folder
        if not Path(row["audio"]).is_absolute():
            row["audio"] = str(Path(path).parent / row["audio"])
    return rows
