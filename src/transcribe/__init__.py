'''
English-Swahili Code-Switching Transcriber

A custom speech-to-text transcriber designed for English-Swahili code-switching
speech patterns commonly found in Kenya.
'''

__version__ = '0.1.0'
__author__ = 'Transcriber Team'

# Heavy submodules (torch, librosa) are imported lazily so lightweight tools
# such as the pilot and the transcript parsers work without loading them.
_LAZY = {
    'load_config': ('.legacy.config', 'load_config'),
    'Config': ('.legacy.config', 'Config'),
    'CTCCodeSwitchingTranscriber': ('.legacy.models.ctc_model', 'CTCCodeSwitchingTranscriber'),
    'Seq2SeqCodeSwitchingTranscriber': ('.legacy.models.seq2seq', 'Seq2SeqCodeSwitchingTranscriber'),
    'DataPreprocessor': ('.legacy.data.preprocessing', 'DataPreprocessor'),
    'EvaluationMetrics': ('.legacy.evaluation.metrics', 'EvaluationMetrics'),
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name, __name__), attr)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
