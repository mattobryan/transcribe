'''
Evaluation metrics: WER, CER, BLEU with proper token handling.
'''

import numpy as np
import torch


def _editdistance(a, b) -> int:
    '''
    Compute Levenshtein edit distance efficiently.
    Falls back gracefully if editdistance package is missing.
    '''
    try:
        import editdistance as ed
        return ed.eval(a, b)
    except ImportError:
        import Levenshtein
        if isinstance(a, str) and isinstance(b, str):
            return Levenshtein.distance(a, b)
        # For lists: fall back to dynamic programming
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]
                    )
        return dp[m][n]


class EvaluationMetrics:
    '''
    Provides WER, CER, and BLEU evaluation metrics for ASR.
    '''

    def __init__(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.special_tokens = {'<PAD>', '<UNK>', '<SOS>', '<EOS>'}

    def indices_to_text(self, indices) -> str:
        '''
        Convert indices to text, stripping special tokens and stopping at EOS.
        '''
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        text = ''
        for idx in indices:
            idx = int(idx)
            if idx not in self.idx_to_char:
                continue
            char = self.idx_to_char[idx]
            if char == '<EOS>':
                break
            if char in self.special_tokens:
                continue
            text += char if char != ' ' else ' '
        return text

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        '''
        Word Error Rate.
        '''
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0

        distance = _editdistance(ref_words, hyp_words)
        return distance / len(ref_words)

    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        '''
        Character Error Rate.
        '''
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        distance = _editdistance(reference, hypothesis)
        return distance / len(reference)

    def calculate_bleu(self, references, hypotheses) -> float:
        '''
        BLEU score (character-level, using nltk if available).
        '''
        try:
            from nltk.translate.bleu_score import sentence_bleu
            bleu_scores = []
            for ref, hyp in zip(references, hypotheses):
                try:
                    ref_chars = [list(ref)]
                    hyp_chars = list(hyp)
                    score = sentence_bleu(ref_chars, hyp_chars)
                    bleu_scores.append(score)
                except Exception:
                    bleu_scores.append(0.0)
            return np.mean(bleu_scores) if bleu_scores else 0.0
        except (ImportError, LookupError):
            # Fallback: simple n-gram precision
            return self._simple_n_gram_bleu(references, hypotheses)

    def _simple_n_gram_bleu(self, references, hypotheses) -> float:
        '''Simple fallback BLEU using character n-gram precision.'''
        scores = []
        for ref, hyp in zip(references, hypotheses):
            if not hyp:
                scores.append(0.0)
                continue
            ref_set = set(zip(ref, ref[1:]))
            hyp_ngrams = list(zip(hyp, hyp[1:]))
            if not hyp_ngrams:
                scores.append(0.0)
                continue
            matches = sum(1 for ng in hyp_ngrams if ng in ref_set)
            precision = matches / len(hyp_ngrams)
            scores.append(precision)
        return np.mean(scores) if scores else 0.0

    def evaluate_text_pairs(self, references, hypotheses) -> dict:
        '''
        Evaluate from lists of reference/hypothesis strings.

        Returns:
            {'wer': float, 'cer': float, 'bleu': float, 'num_samples': int}
        '''
        wer_scores = [
            self.calculate_wer(r, h)
            for r, h in zip(references, hypotheses)
        ]
        cer_scores = [
            self.calculate_cer(r, h)
            for r, h in zip(references, hypotheses)
        ]

        return {
            'wer': float(np.mean(wer_scores)) if wer_scores else 1.0,
            'cer': float(np.mean(cer_scores)) if cer_scores else 1.0,
            'bleu': float(self.calculate_bleu(references, hypotheses)),
            'num_samples': len(references),
        }


if __name__ == '__main__':
    metrics = EvaluationMetrics(
        char_to_idx={}, idx_to_char={0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
    )
    # Character mapping is irrelevant for string-based tests
    refs = ['hello world', 'jambo dunia']
    hyps = ['hellow world', 'jambo']
    results = metrics.evaluate_text_pairs(refs, hyps)
    print(f'Metrics: {results}')
