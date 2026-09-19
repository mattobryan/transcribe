'''
Data preprocessing utilities: audio loading, vocabulary, tokenization,
and dataset splitting.
'''

import os
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
from sklearn.model_selection import train_test_split


SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']


class DataPreprocessor:
    '''
    Handles audio loading, MFCC feature extraction, vocabulary creation,
    text tokenization, and dataset splitting.
    '''

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_audio_seconds: Optional[float] = None,
        max_seq_length: int = 200,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_audio_seconds = max_audio_seconds
        self.max_seq_length = max_seq_length

    def load_audio(self, file_path) -> np.ndarray:
        '''
        Load audio file and convert to MFCC features.

        Returns:
            MFCC features of shape (seq_len, n_mfcc)
        '''
        audio, sr = librosa.load(file_path, sr=self.sample_rate)

        # Truncate audio if too long
        if self.max_audio_seconds is not None:
            max_samples = int(self.max_audio_seconds * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return mfccs.T

    def create_vocabulary(
        self, transcripts: List[str], min_char_frequency: int = 1
    ) -> List[str]:
        '''
        Create vocabulary from transcripts, filtering by character frequency.
        Preserves case to handle proper nouns in Swahili/English.

        Returns:
            Sorted list of characters (vocabulary)
        '''
        char_counts = Counter()
        for transcript in transcripts:
            char_counts.update(transcript)

        # Filter by minimum frequency
        vocab = [
            char for char, count in char_counts.items()
            if count >= min_char_frequency and char != ' '
        ]

        # Special tokens first (indices 0-3)
        return SPECIAL_TOKENS + sorted(vocab)

    def build_mappings(
        self, vocab: List[str]
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        '''Build char_to_idx and idx_to_char mappings from vocabulary.'''
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        return char_to_idx, idx_to_char

    def text_to_indices(
        self,
        text: str,
        char_to_idx: Dict[str, int],
        add_sos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        '''
        Convert text to sequence of character indices.
        '''
        indices = []
        if add_sos:
            indices.append(char_to_idx['<SOS>'])
        for char in text.replace(' ', ''):
            indices.append(char_to_idx.get(char, char_to_idx['<UNK>']))
        if add_eos:
            indices.append(char_to_idx['<EOS>'])
        return indices

    def indices_to_text(
        self, indices, idx_to_char: Dict[int, str]
    ) -> str:
        '''
        Convert index sequence back to text.
        '''
        text = ''
        for idx in indices:
            if idx not in idx_to_char:
                continue
            char = idx_to_char[idx]
            if char in SPECIAL_TOKENS:
                continue
            text += char
        return text

    def prepare_dataset(
        self,
        data_dir: str,
        metadata_file: str,
        val_split: float = 0.1,
        min_char_frequency: int = 1,
        random_state: int = 42,
    ) -> Dict[str, object]:
        '''
        Prepare dataset from directory of audio files and metadata JSON.

        Args:
            data_dir: Directory containing audio files
            metadata_file: JSON file with [{audio_file, transcript}, ...]
            val_split: Fraction for validation
            min_char_frequency: Minimum char frequency for vocabulary
            random_state: Random seed

        Returns:
            Dictionary with train/val/test splits, vocab, and mappings
        '''
        data_dir = Path(data_dir)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        audio_files = []
        transcripts = []

        for item in metadata:
            audio_path = data_dir / item['audio_file']
            if audio_path.exists():
                audio_files.append(str(audio_path))
                transcripts.append(item['transcript'])

        if len(audio_files) == 0:
            raise ValueError(
                f'No valid audio files found in {data_dir}. '
                f'Metadata: {metadata_file}'
            )

        # Create vocabulary and mappings
        vocab = self.create_vocabulary(transcripts, min_char_frequency)
        char_to_idx, idx_to_char = self.build_mappings(vocab)

        # Convert transcripts to indices
        indexed_transcripts = [
            self.text_to_indices(t, char_to_idx) for t in transcripts
        ]

        # Split dataset
        X_tr, X_te, y_tr, y_te = train_test_split(
            audio_files, indexed_transcripts,
            test_size=0.2, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=val_split, random_state=random_state
        )

        return {
            'train': {'audio': X_train, 'transcripts': y_train},
            'val': {'audio': X_val, 'transcripts': y_val},
            'test': {'audio': X_te, 'transcripts': y_te},
            'vocab': vocab,
            'char_to_idx': char_to_idx,
            'idx_to_char': idx_to_char,
        }


if __name__ == '__main__':
    preprocessor = DataPreprocessor()
    print('DataPreprocessor initialized')

    # Vocabulary test
    v = preprocessor.create_vocabulary(['hello', 'jambo', 'Habari'])
    print(f'Vocabulary: {v}')
    c2i, i2c = preprocessor.build_mappings(v)
    idx = preprocessor.text_to_indices('hello', c2i)
    print(f'Text to indices: {idx}')
    text = preprocessor.indices_to_text(idx, i2c)
    print(f'Back to text: {text}')
