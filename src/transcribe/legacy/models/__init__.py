"""
Model architectures for code-switching speech recognition.
"""

from .ctc_model import CTCCodeSwitchingTranscriber
from .seq2seq import Seq2SeqCodeSwitchingTranscriber

__all__ = ["CTCCodeSwitchingTranscriber", "Seq2SeqCodeSwitchingTranscriber"]