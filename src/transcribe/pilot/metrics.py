"""WER, CER, per-language WER and switch-point error rate (TRD section 8.3)."""

from collections import Counter
from typing import Dict, List, Sequence

import jiwer

from .textprep import normalize_text

CODE_LANGS = ("en", "sw")


def switch_points(langs: Sequence[str]) -> List[bool]:
    """True for words next to an en/sw change, and for intra-word mixed words."""
    marks = []
    for i, lang in enumerate(langs):
        if lang == "mixed":
            marks.append(True)
            continue
        neighbours = [langs[j] for j in (i - 1, i + 1) if 0 <= j < len(langs)]
        marks.append(lang in CODE_LANGS and any(n in CODE_LANGS and n != lang for n in neighbours))
    return marks


def score_segment(ref_words: List[str], ref_langs: List[str], hypothesis: str) -> Dict:
    hyp_words = normalize_text(hypothesis)
    ref = " ".join(ref_words)
    hyp = " ".join(hyp_words)
    result = {"n_ref": len(ref_words), "hyp_norm": hyp}
    if not ref_words:
        result.update(errors=len(hyp_words), subs=0, dels=0, ins=len(hyp_words),
                      char_errors=len(hyp), n_chars=0, word_errors=[], pairs=[])
        return result
    if not hyp_words:
        word_errors = [True] * len(ref_words)
        result.update(errors=len(ref_words), subs=0, dels=len(ref_words), ins=0,
                      char_errors=len(ref), n_chars=len(ref), pairs=[])
    else:
        out = jiwer.process_words(ref, hyp)
        word_errors = [False] * len(ref_words)
        pairs = []
        for chunk in out.alignments[0]:
            if chunk.type in ("substitute", "delete"):
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    word_errors[i] = True
            if chunk.type == "substitute":
                ref_span = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
                hyp_span = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                pairs.extend(zip(ref_span, hyp_span))
        chars = jiwer.process_characters(ref, hyp)
        result.update(errors=out.substitutions + out.deletions + out.insertions,
                      subs=out.substitutions, dels=out.deletions, ins=out.insertions,
                      char_errors=chars.substitutions + chars.deletions + chars.insertions,
                      n_chars=len(ref), pairs=pairs)
    result["word_errors"] = word_errors
    return result


def aggregate(scored: List[Dict], segments: List[Dict]) -> Dict:
    """Micro-averaged metrics over segments (errors / reference words)."""
    totals = Counter()
    lang_err, lang_n = Counter(), Counter()
    sp_err = sp_n = 0
    pairs = Counter()
    for score, segment in zip(scored, segments):
        totals["errors"] += score["errors"]
        totals["subs"] += score["subs"]
        totals["dels"] += score["dels"]
        totals["ins"] += score["ins"]
        totals["n_ref"] += score["n_ref"]
        totals["char_errors"] += score["char_errors"]
        totals["n_chars"] += score["n_chars"]
        langs = segment["ref_langs"]
        for error, lang in zip(score["word_errors"], langs):
            lang_n[lang] += 1
            lang_err[lang] += int(error)
        for error, is_switch in zip(score["word_errors"], switch_points(langs)):
            if is_switch:
                sp_n += 1
                sp_err += int(error)
        pairs.update(score.get("pairs", []))

    def ratio(a, b):
        return round(a / b, 4) if b else None

    return {
        "segments": len(scored),
        "ref_words": totals["n_ref"],
        "wer": ratio(totals["errors"], totals["n_ref"]),
        "cer": ratio(totals["char_errors"], totals["n_chars"]),
        "substitutions": totals["subs"], "deletions": totals["dels"], "insertions": totals["ins"],
        "wer_by_lang": {lang: ratio(lang_err[lang], lang_n[lang]) for lang in sorted(lang_n)},
        "words_by_lang": dict(sorted(lang_n.items())),
        "switch_point_words": sp_n,
        "switch_point_error_rate": ratio(sp_err, sp_n),
        "top_substitutions": [[r, h, c] for (r, h), c in pairs.most_common(25)],
    }
