import os
import json
import librosa
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def load_audio(self, file_path):
        """Load audio file and convert to MFCC features"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        return mfccs.T
    
    def create_vocabulary(self, transcripts):
        """Create vocabulary from transcripts including English and Swahili characters"""
        chars = set()
        for transcript in transcripts:
            chars.update(transcript.lower())
        # Add special tokens
        chars.add('<PAD>')
        chars.add('<UNK>')
        chars.add('<SOS>')
        chars.add('<EOS>')
        return sorted(list(chars))
    
    def text_to_indices(self, text, char_to_idx):
        """Convert text to sequence of indices"""
        indices = [char_to_idx.get('<SOS>')]
        for char in text.lower():
            indices.append(char_to_idx.get(char, char_to_idx.get('<UNK>')))
        indices.append(char_to_idx.get('<EOS>'))
        return indices
    
    def prepare_dataset(self, data_dir, metadata_file):
        """Prepare dataset from directory of audio files and metadata"""
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        audio_files = []
        transcripts = []
        
        for item in metadata:
            audio_path = os.path.join(data_dir, item['audio_file'])
            if os.path.exists(audio_path):
                audio_files.append(audio_path)
                transcripts.append(item['transcript'])
        
        # Create vocabulary
        vocab = self.create_vocabulary(transcripts)
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        
        # Convert transcripts to indices
        indexed_transcripts = [self.text_to_indices(t, char_to_idx) for t in transcripts]
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            audio_files, indexed_transcripts, test_size=0.2, random_state=42
        )
        
        return {
            'train': {'audio': X_train, 'transcripts': y_train},
            'test': {'audio': X_test, 'transcripts': y_test},
            'vocab': vocab,
            'char_to_idx': char_to_idx
        }

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    print("DataPreprocessor initialized")
