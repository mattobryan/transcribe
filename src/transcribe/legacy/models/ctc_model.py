'''
CNN-LSTM with CTC loss for code-switching speech recognition.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CTCCodeSwitchingTranscriber(nn.Module):
    '''
    CNN-LSTM transducer with CTC output for speech recognition.
    
    Architecture:
    - CNN feature extractor (3 Conv1D layers with pooling) - reduces time by 8x
    - Bidirectional LSTM layers for sequence modeling
    - Linear projection to vocabulary size (per-frame logits)
    - CTC loss for training (handles alignment)
    
    Output shape: (batch, seq_len, vocab_size) - logits for each timestep
    '''

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 256,
        num_layers: int = 3,
        vocab_size: int = 50,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # CNN feature extractor - reduces sequence length by factor of 8
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        cnn_out_channels = 256

        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output projection (per-frame)
        lstm_out_dim = hidden_dim * self.num_directions
        self.output_proj = nn.Linear(lstm_out_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        '''Initialize model weights.'''
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            input_lengths: Optional tensor of actual sequence lengths (batch,)

        Returns:
            log_probs: Log probabilities of shape (batch, seq_len, vocab_size)
            output_lengths: Actual output sequence lengths after CNN pooling
        '''
        batch_size = x.size(0)

        # CNN feature extraction
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.cnn(x)         # (batch, 256, seq_len//8)
        x = x.transpose(1, 2)   # (batch, seq_len//8, 256)

        # Calculate output lengths after CNN pooling (3x MaxPool1d(2) = /8)
        if input_lengths is not None:
            output_lengths = input_lengths // 8
            output_lengths = torch.clamp(output_lengths, min=1)
        else:
            output_lengths = torch.full(
                (batch_size,), x.size(1),
                dtype=torch.long, device=x.device
            )

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Output projection (per-frame logits)
        logits = self.output_proj(lstm_out)

        # CTC expects log_probs
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, output_lengths

    def get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        '''Calculate output sequence lengths after CNN pooling.'''
        return torch.clamp(input_lengths // 8, min=1)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        blank_idx: int = 0
    ) -> list:
        '''
        Greedy CTC decoding for inference.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            input_lengths: Optional input lengths
            blank_idx: Index of blank token (usually 0 for PAD)

        Returns:
            List of decoded token sequences (without blanks/repeats)
        '''
        self.eval()
        log_probs, output_lengths = self.forward(x, input_lengths)

        # Greedy decoding
        predictions = log_probs.argmax(dim=-1)

        decoded_sequences = []
        for i in range(predictions.size(0)):
            seq = predictions[i, :output_lengths[i]].cpu().numpy()
            # CTC greedy decode: remove blanks and consecutive duplicates
            decoded = []
            prev = blank_idx
            for token in seq:
                if token != blank_idx and token != prev:
                    decoded.append(int(token))
                prev = token
            decoded_sequences.append(decoded)

        return decoded_sequences

    @torch.no_grad()
    def predict_beam_search(
        self,
        x: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        blank_idx: int = 0,
        beam_width: int = 10
    ) -> list:
        '''Beam search CTC decoding - falls back to greedy for now.'''
        return self.predict(x, input_lengths, blank_idx)

    def num_parameters(self) -> int:
        '''Return total number of trainable parameters.'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = CTCCodeSwitchingTranscriber(vocab_size=50)
    print(f'CTC Model initialized with {model.num_parameters()} parameters')

    # Quick test
    batch_size, seq_len, input_dim = 2, 1600, 13
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    dummy_lengths = torch.tensor([1600, 1500])
    log_probs, out_lens = model(dummy_input, dummy_lengths)
    print(f'Output shape: {log_probs.shape}')
    print(f'Output lengths: {out_lens}')
    preds = model.predict(dummy_input, dummy_lengths)
    print(f'Predictions: {preds}')
