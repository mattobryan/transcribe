'''
English-Swahili Code-Switching Transcriber

A custom speech-to-text transcriber designed for English-Swahili code-switching
speech patterns commonly found in Kenya.
'''

__version__ = '0.1.0'
__author__ = 'Transcriber Team'

from .config import load_config, Config
from .models.ctc_model import CTCCodeSwitchingTranscriber
from .models.seq2seq import Seq2SeqCodeSwitchingTranscriber
from .data.preprocessing import DataPreprocessor
from .evaluation.metrics import EvaluationMetrics

__all__ = [
    'load_config',
    'Config',
    'CTCCodeSwitchingTranscriber',
    'Seq2SeqCodeSwitchingTranscriber',
    'DataPreprocessor',
    'EvaluationMetrics',
]
