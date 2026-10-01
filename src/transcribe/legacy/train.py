'''
Training entry point.

Usage:
    python -m transcribe.legacy.train --config config/default.yaml
'''

import argparse
import pickle

from torch.utils.data import DataLoader


from transcribe.legacy.config import load_config
from transcribe.legacy.data.preprocessing import DataPreprocessor
from transcribe.legacy.data.dataset import SpeechDataset, collate_ctc
from transcribe.legacy.models.ctc_model import CTCCodeSwitchingTranscriber
from transcribe.legacy.models.seq2seq import Seq2SeqCodeSwitchingTranscriber
from transcribe.legacy.training.trainer import Trainer


def build_model(config, vocab_size):
    if config.model_type == 'ctc':
        return CTCCodeSwitchingTranscriber(
            input_dim=config.n_mfcc,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            vocab_size=vocab_size,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        )
    else:
        return Seq2SeqCodeSwitchingTranscriber(
            input_dim=config.n_mfcc,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            vocab_size=vocab_size,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            decoder_layers=config.decoder_layers,
            teacher_forcing_ratio=config.teacher_forcing_ratio,
        )


def main():
    parser = argparse.ArgumentParser(description='Train transcriber')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f'Device: {config.device}')
    print(f'Model type: {config.model_type}')

    preprocessor = DataPreprocessor(
        sample_rate=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        max_audio_seconds=config.max_audio_length,
        max_seq_length=config.max_sequence_length,
    )

    print('Preparing dataset...')
    dataset = preprocessor.prepare_dataset(
        data_dir=str(config.data_dir),
        metadata_file=str(config.metadata_file),
        min_char_frequency=config.min_char_frequency,
    )

    vocab_size = len(dataset['vocab'])
    print(f'Vocabulary size: {vocab_size}')

    train_dataset = SpeechDataset(
        dataset['train']['audio'],
        dataset['train']['transcripts'],
        preprocessor,
    )
    val_dataset = SpeechDataset(
        dataset['val']['audio'],
        dataset['val']['transcripts'],
        preprocessor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_ctc,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_ctc,
        num_workers=0,
    )

    model = build_model(config, vocab_size)
    param_count = model.num_parameters() if hasattr(model, 'num_parameters') else 0
    print(f'Model parameters: {param_count}')

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(config.checkpoint_dir / 'vocab.pkl', 'wb') as f:
        pickle.dump({
            'vocab': dataset['vocab'],
            'char_to_idx': dataset['char_to_idx'],
            'idx_to_char': dataset['idx_to_char'],
        }, f)

    trainer = Trainer(
        model, config, train_loader, val_loader, device=config.device,
    )
    trainer.train()


if __name__ == '__main__':
    main()
