"""Learn from a reviewer's corrections while they work, without retraining.

Three things are derived from the chunks already corrected in a session:

* **Correction rules.** A one-word substitution the reviewer made at least
  ``MIN_COUNT`` times (``sababa`` -> ``sababu``), and never left uncorrected more
  often than corrected, is applied to every chunk not yet edited.
* **Prompt.** The most recent corrected text, passed to Whisper as
  ``initial_prompt`` so spellings and code-switching style carry forward.
* **Hotwords.** Words the reviewer typed that the model never produced (names,
  places, hybrid spellings), passed to Whisper as ``hotwords``.

Model weights are updated separately, in batches, from all corrected chunks
(Implementation Plan, Phase 4).
"""

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import jiwer

MIN_COUNT = 2
PROMPT_CHARS = 220
MAX_HOTWORDS = 40
TOKEN = re.compile(r"\S+")


def _norm(word: str) -> str:
    return re.sub(r"[^\w'-]", "", word.lower()).strip("'-")


def word_pairs(model_text: str, corrected: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """One-to-one substitutions made by the reviewer, and model words they kept."""
    hyp = [_norm(w) for w in TOKEN.findall(model_text)]
    ref = [_norm(w) for w in TOKEN.findall(corrected)]
    hyp, ref = [w for w in hyp if w], [w for w in ref if w]
    if not hyp or not ref:
        return [], []
    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    pairs, kept = [], []
    for chunk in out.alignments[0]:
        if chunk.type == "substitute" and (chunk.ref_end_idx - chunk.ref_start_idx) == (
                chunk.hyp_end_idx - chunk.hyp_start_idx):
            pairs += zip(hyp[chunk.hyp_start_idx:chunk.hyp_end_idx], ref[chunk.ref_start_idx:chunk.ref_end_idx])
        elif chunk.type == "equal":
            kept += hyp[chunk.hyp_start_idx:chunk.hyp_end_idx]
    return pairs, kept


def learn(segments: Iterable[Dict]) -> Dict:
    """Rules, prompt and hotwords from the edited segments, in order."""
    fixes: Counter = Counter()
    kept: Counter = Counter()
    model_words: set = set()
    typed: List[str] = []
    corrected_texts: List[str] = []
    for seg in segments:
        model_text = seg.get("asr_hypothesis") or ""
        model_words.update(_norm(w) for w in TOKEN.findall(model_text))
        if not seg.get("edited"):
            continue
        text = seg.get("transcript") or ""
        corrected_texts.append(text)
        pairs, same = word_pairs(model_text, text)
        fixes.update(pairs)
        kept.update(same)
        typed += [w for w in TOKEN.findall(text)]
    rules = []
    for (wrong, right), count in fixes.most_common():
        if count >= MIN_COUNT and wrong != right and count > kept[wrong]:
            if not any(r[0] == wrong for r in rules):           # keep the most frequent fix per word
                rules.append([wrong, right, count])
    hotwords, seen = [], set()
    for word in typed:
        key = _norm(word)
        clean = word.strip(".,;:!?\"()")
        if key and key not in model_words and key not in seen and len(key) > 2:
            seen.add(key)
            hotwords.append(clean)
    prompt = " ".join(corrected_texts)[-PROMPT_CHARS:]
    if " " in prompt:
        prompt = prompt[prompt.index(" ") + 1:]                 # do not start mid-word
    return {"rules": rules, "prompt": prompt.strip(), "hotwords": hotwords[-MAX_HOTWORDS:]}


def apply_rules(text: str, rules: List[List]) -> str:
    """Apply learned word substitutions, keeping the original punctuation and capitalisation style."""
    if not rules or not text:
        return text
    table = {wrong: right for wrong, right, _ in rules}

    def swap(match: re.Match) -> str:
        token = match.group()
        core = _norm(token)
        if core not in table:
            return token
        lead = re.match(r"^\W*", token).group()
        trail = re.search(r"\W*$", token).group()
        right = table[core]
        if token[len(lead):len(lead) + 1].isupper():
            right = right[:1].upper() + right[1:]
        return lead + right + trail

    return TOKEN.sub(swap, text)
