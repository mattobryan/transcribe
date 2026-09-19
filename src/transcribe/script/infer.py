'''
Single-file inference entry point.

Usage:
    python -m src.transcribe.script.infer --audio path/to/file.wav \
        --checkpoint checkpoints/final_model.pt
'''

import argparse
import sys
from pathlib import Path
import pickle

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.transcribe.config import load_config
from src.transcribe.data.preprocessing import DataPreprocessor
from src.transcribe.models.ctc_model import CTCCodeSwitchingTranscriber
from src.transcribe.models.seq2seq import Seq2SeqCodeSwitchingTranscriber


def load_checkpoint(path):
    '''Load checkpoint and return model, config, vocab mappings.'''
    ckpt = torch.load(path, map_location='cpu')
    ckpt_path = Path(path)
    with open(ckpt_path.parent / 'vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    return ckpt, vocab_data


def build_model_from_ckpt(ckpt, vocab_data):
    config_dict = ckpt.get('config', {})
    model_type = ckpt.get('model_type', config_dict.get('model_type', 'ctc'))
    vocab_size = len(vocab_data['vocab'])
    input_dim = config_dict.get('n_mfcc', 13)
    hidden_dim = config_dict.get('hidden_dim', 256)
    num_layers = config_dict.get('num_layers', 3)
    dropout = config_dict.get('dropout', 0.2)
    bidirectional = config_dict.get('bidirectional', True)

    if model_type == 'ctc':
        return CTCCodeSwitchingTranscriber(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, vocab_size=vocab_size,
            dropout=dropout, bidirectional=bidirectional,
        )
    else:
        return Seq2SeqCodeSwitchingTranscriber(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, vocab_size=vocab_size,
            dropout=dropout, bidirectional=bidirectional,
            decoder_layers=config_dict.get('decoder_layers', 2),
        )


def main():
    parser = argparse.ArgumentParser(description='Infer transcription')
    parser.add_argument('--audio', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt, vocab_data = load_checkpoint(args.checkpoint)

    model = build_model_from_ckpt(ckpt, vocab_data)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preprocessor = DataPreprocessor(
        sample_rate=config.sample_rate, n_mfcc=config.n_mfcc,
        n_fft=config.n_fft, hop_length=config.hop_length,
        max_audio_seconds=config.max_audio_length,
    )

    mfccs = preprocessor.load_audio(args.audio)
    input_tensor = torch.FloatTensor(mfccs).unsqueeze(0)  # (1, seq, dim)
    input_len = torch.tensor([input_tensor.size(1)])

    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}

    model_type = ckpt.get('model_type', 'ctc')
    if model_type == 'ctc':
        decoded = model.predict(input_tensor, input_len, blank_idx=0)
        token_seq = decoded[0]
        text = ''.join(idx_to_char.get(int(t), '') for t in token_seq)
    else:
        sos = vocab_data['char_to_idx']['<SOS>']
        eos = vocab_data['char_to_idx']['<EOS>']
        decoded = model.predict(
            input_tensor, input_len, sos_idx=sos, eos_idx=eos
        )
        token_seq = decoded[0]
        text = ''
        for t in token_seq:
            if int(t) == eos:
                break
            text += idx_to_char.get(int(t), '')

    print(f'Transcription: {text}')


if __name__ == '__main__':
    main()
