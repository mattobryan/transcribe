'''
Evaluation entry point.

Usage:
    python -m src.transcribe.script.evaluate --checkpoint checkpoints/final_model.pt \
        --config config/default.yaml
'''

import argparse
import sys
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.transcribe.config import load_config
from src.transcribe.data.preprocessing import DataPreprocessor
from src.transcribe.data.dataset import SpeechDataset, collate_ctc
from src.transcribe.models.ctc_model import CTCCodeSwitchingTranscriber
from src.transcribe.models.seq2seq import Seq2SeqCodeSwitchingTranscriber
from src.transcribe.evaluation.metrics import EvaluationMetrics


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    with open(Path(path).parent / 'vocab.pkl', 'rb') as f:
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
    parser = argparse.ArgumentParser(description='Evaluate transcriber')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt, vocab_data = load_checkpoint(args.checkpoint)

    model = build_model_from_ckpt(ckpt, vocab_data)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(config.device)
    model.eval()

    preprocessor = DataPreprocessor(
        sample_rate=config.sample_rate, n_mfcc=config.n_mfcc,
        n_fft=config.n_fft, hop_length=config.hop_length,
    )

    dataset = preprocessor.prepare_dataset(
        data_dir=str(config.data_dir),
        metadata_file=str(config.metadata_file),
    )

    test_dataset = SpeechDataset(
        dataset['test']['audio'],
        dataset['test']['transcripts'],
        preprocessor,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_ctc,
    )

    metrics_eval = EvaluationMetrics(
        char_to_idx=dataset['char_to_idx'],
        idx_to_char=dataset['idx_to_char'],
    )

    all_refs, all_hyps = [], []
    with torch.no_grad():
        for batch in test_loader:
            audios, transcripts, audio_lens, trans_lens = batch
            audios = audios.to(config.device)
            audio_lens = audio_lens.to(config.device)

            model_type = ckpt.get('model_type', 'ctc')
            if model_type == 'ctc':
                decoded = model.predict(audios, audio_lens)
            else:
                sos = vocab_data['char_to_idx']['<SOS>']
                eos = vocab_data['char_to_idx']['<EOS>']
                decoded = model.predict(audios, audio_lens, sos, eos)

            for i in range(len(decoded)):
                ref_text = metrics_eval.indices_to_text(
                    transcripts[i].numpy(),
                )
                pred_text = ''.join(
                    vocab_data['idx_to_char'].get(int(t), '')
                    for t in decoded[i]
                )
                all_refs.append(ref_text)
                all_hyps.append(pred_text)

    results = metrics_eval.evaluate_text_pairs(all_refs, all_hyps)
    print(f'Validation Results: {results}')


if __name__ == '__main__':
    main()
