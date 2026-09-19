'''
PyTorch Dataset and DataLoader utilities.
'''

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class SpeechDataset(Dataset):
    '''
    PyTorch dataset for code-switching speech recognition.
    Loads audio, extracts MFCC features, and tokenizes transcripts.
    '''

    def __init__(
        self,
        audio_files: List[str],
        transcripts: List[list],
        preprocessor,
    ):
        '''
        Args:
            audio_files: List of paths to audio files
            transcripts: List of pre-indexed transcript token sequences
            preprocessor: DataPreprocessor instance
        '''
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio and extract MFCCs
        mfccs = self.preprocessor.load_audio(self.audio_files[idx])

        # Get transcript token indices
        transcript = self.transcripts[idx]

        return torch.FloatTensor(mfccs), torch.LongTensor(transcript)


def collate_ctc(batch):
    '''
    Collate function for CTC model training.
    Returns padded audios, transcripts, audio_lengths, transcript_lengths.
    '''
    audios, transcripts = zip(*batch)

    # Pad audio sequences
    audio_lengths = [a.size(0) for a in audios]
    max_audio_len = max(audio_lengths)
    feat_dim = audios[0].size(1) if audios else 1

    padded_audios = torch.zeros(len(audios), max_audio_len, feat_dim)
    for i, audio in enumerate(audios):
        padded_audios[i, :audio.size(0)] = audio

    # Pad transcripts
    trans_lengths = [t.size(0) for t in transcripts]
    max_trans_len = max(trans_lengths)
    padded_transcripts = torch.zeros(
        len(transcripts), max_trans_len, dtype=torch.long
    )
    for i, transcript in enumerate(transcripts):
        padded_transcripts[i, :transcript.size(0)] = transcript

    audio_lengths_t = torch.tensor(audio_lengths, dtype=torch.long)
    trans_lengths_t = torch.tensor(trans_lengths, dtype=torch.long)

    return (
        padded_audios,
        padded_transcripts,
        audio_lengths_t,
        trans_lengths_t,
    )


def collate_seq2seq(batch):
    '''
    Collate function for seq2seq model training.
    Same as collate_ctc but strips SOS/EOS appropriately handled in trainer.
    '''
    return collate_ctc(batch)
