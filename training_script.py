import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from model_architecture import CodeSwitchingTranscriber
from data_preprocessing import DataPreprocessor

class SpeechDataset(Dataset):
    def __init__(self, audio_files, transcripts, preprocessor):
        self.audio_files = audio_files
        self.transcripts = transcripts
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load and preprocess audio
        mfccs = self.preprocessor.load_audio(self.audio_files[idx])
        
        # Get transcript
        transcript = self.transcripts[idx]
        
        return torch.FloatTensor(mfccs), torch.LongTensor(transcript)

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    audios, transcripts = zip(*batch)
    
    # Pad audio sequences
    audio_lengths = [len(audio) for audio in audios]
    max_audio_len = max(audio_lengths)
    padded_audios = torch.zeros(len(audios), max_audio_len, audios[0].size(1))
    for i, audio in enumerate(audios):
        padded_audios[i, :len(audio)] = audio
    
    # Pad transcripts
    transcript_lengths = [len(transcript) for transcript in transcripts]
    max_transcript_len = max(transcript_lengths)
    padded_transcripts = torch.zeros(len(transcripts), max_transcript_len, dtype=torch.long)
    for i, transcript in enumerate(transcripts):
        padded_transcripts[i, :len(transcript)] = transcript
    
    return padded_audios, padded_transcripts, audio_lengths, transcript_lengths

def train_model(config):
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        sample_rate=config['sample_rate'],
        n_mfcc=config['n_mfcc']
    )
    
    # Prepare dataset
    dataset = preprocessor.prepare_dataset(
        config['data_dir'],
        config['metadata_file']
    )
    
    # Create datasets
    train_dataset = SpeechDataset(
        dataset['train']['audio'],
        dataset['train']['transcripts'],
        preprocessor
    )
    
    test_dataset = SpeechDataset(
        dataset['test']['audio'],
        dataset['test']['transcripts'],
        preprocessor
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = CodeSwitchingTranscriber(
        input_dim=config['n_mfcc'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        vocab_size=len(dataset['vocab']),
        dropout=config['dropout']
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (audios, transcripts, audio_lens, trans_lens) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(audios)
            
            # Calculate loss (simplified - in practice you'd need proper alignment)
            loss = criterion(outputs, transcripts[:, -1])  # Last token prediction
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{config["epochs"]}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab': dataset['vocab']
            }, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dataset['vocab'],
        'config': config
    }, os.path.join(config['checkpoint_dir'], 'final_model.pth'))
    
    print("Training completed!")

if __name__ == "__main__":
    # Default configuration
    config = {
        'data_dir': './data/audio',
        'metadata_file': './data/metadata.json',
        'checkpoint_dir': './checkpoints',
        'sample_rate': 16000,
        'n_mfcc': 13,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.2,
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 100,
        'save_every': 10
    }
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    train_model(config)
