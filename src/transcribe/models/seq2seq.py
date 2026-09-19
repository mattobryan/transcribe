'''
Encoder-Decoder with attention for code-switching speech recognition.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class Seq2SeqCodeSwitchingTranscriber(nn.Module):
    '''
    Encoder-Decoder architecture with attention for speech recognition.
    
    Architecture:
    - Encoder: CNN + BiLSTM for audio feature extraction
    - Decoder: LSTM with attention over encoder outputs, autoregressive generation
    - Teacher forcing during training, greedy/beam search during inference
    
    This model outputs per-timestep predictions and uses CrossEntropyLoss
    for training (requires aligned target sequences).
    '''

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 256,
        num_layers: int = 3,
        vocab_size: int = 50,
        dropout: float = 0.2,
        bidirectional: bool = True,
        decoder_layers: int = 2,
        teacher_forcing_ratio: float = 0.5,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.decoder_layers = decoder_layers

        # CNN feature extractor
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
        encoder_out_dim = hidden_dim * self.num_directions

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Project encoder hidden states to decoder dimension
        self.encoder_hidden_proj = nn.Linear(encoder_out_dim, hidden_dim)
        self.encoder_cell_proj = nn.Linear(encoder_out_dim, hidden_dim)

        # Decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Project concatenated (context + embedding) to decoder input size
        self.decoder_input_proj = nn.Linear(encoder_out_dim + hidden_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,  # embedding dim
            hidden_size=hidden_dim,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dropout if decoder_layers > 1 else 0,
            bidirectional=False,
        )

        # Attention mechanism (additive/Luong style)
        self.attention = nn.Sequential(
            nn.Linear(encoder_out_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(encoder_out_dim + hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def encode(
        self, src: torch.Tensor, src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Encode audio features.

        Returns:
            encoder_out: (batch, seq_len, encoder_out_dim)
            encoder_lengths: (batch,) actual encoder output lengths
        '''
        batch_size = src.size(0)

        # CNN feature extraction
        x = src.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.cnn(x)           # (batch, 256, seq_len//8)
        x = x.transpose(1, 2)     # (batch, seq_len//8, 256)

        # Calculate output lengths
        if src_lengths is not None:
            encoder_lengths = torch.clamp(src_lengths // 8, min=1)
        else:
            encoder_lengths = torch.full(
                (batch_size,), x.size(1),
                dtype=torch.long, device=x.device
            )

        # LSTM
        encoder_out, _ = self.encoder_lstm(x)

        return encoder_out, encoder_lengths

    def _compute_attention(
        self,
        decoder_hidden: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Compute attention context vector.

        Args:
            decoder_hidden: (batch, hidden_dim)
            encoder_out: (batch, enc_len, encoder_out_dim)
            encoder_mask: (batch, enc_len) - True for valid positions

        Returns:
            context: (batch, encoder_out_dim)
            attn_weights: (batch, enc_len)
        '''
        batch_size, enc_len, _ = encoder_out.shape

        # Expand decoder hidden to all encoder timesteps
        decoder_hidden_exp = decoder_hidden.unsqueeze(1).expand(-1, enc_len, -1)

        # Compute attention scores
        attn_input = torch.cat([encoder_out, decoder_hidden_exp], dim=-1)
        attn_scores = self.attention(attn_input).squeeze(-1)  # (batch, enc_len)

        # Mask out padded positions
        attn_scores = attn_scores.masked_fill(~encoder_mask, float('-inf'))

        # Softmax over encoder positions
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, enc_len)

        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_out).squeeze(1)

        return context, attn_weights

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        tgt_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        Forward pass with teacher forcing.

        Args:
            src: (batch, seq_len, input_dim)
            tgt: (batch, tgt_len) target token indices
            src_lengths: (batch,) input audio lengths
            tgt_lengths: (batch,) target sequence lengths

        Returns:
            outputs: (batch, tgt_len, vocab_size) decoder output logits
        '''
        batch_size, tgt_len, _ = tgt.shape if tgt.dim() == 3 else (tgt.size(0), tgt.size(1), 0)
        tgt_len = tgt.size(1)

        # Encode
        encoder_out, encoder_lengths = self.encode(src, src_lengths)
        encoder_out_dim = encoder_out.size(-1)

        # Build encoder mask
        enc_mask = torch.arange(
            encoder_out.size(1), device=encoder_out.device
        ).unsqueeze(0) < encoder_lengths.unsqueeze(1)

        # Decode step by step
        outputs = []
        prev_token = tgt[:, :-1] if tgt_len > 1 else tgt  # Shift for teacher forcing
        embedded = self.dropout(self.embedding(prev_token))  # (batch, tgt_len, hidden_dim)
        dec_steps = embedded.size(1)

        # Initialize decoder hidden state from encoder
        # Use the last valid encoder output for each sequence
        if src_lengths is not None:
            last_indices = (encoder_lengths - 1).clamp(min=0)
            batch_idx = torch.arange(batch_size, device=encoder_out.device)
            last_enc_out = encoder_out[batch_idx, last_indices]
        else:
            last_enc_out = encoder_out[:, -1, :]

        # Project encoder hidden to decoder hidden.
        # Stack over decoder layers -> (num_layers, batch, hidden_dim)
        decoder_hidden = torch.tanh(self.encoder_hidden_proj(last_enc_out))
        decoder_cell = torch.tanh(self.encoder_cell_proj(last_enc_out))
        decoder_hidden = decoder_hidden.unsqueeze(0).expand(
            self.decoder_layers, *decoder_hidden.shape
        ).contiguous()
        decoder_cell = decoder_cell.unsqueeze(0).expand(
            self.decoder_layers, *decoder_cell.shape
        ).contiguous()

        # Run decoder with attention at each step
        for t in range(dec_steps):
            # Attention
            context, _ = self._compute_attention(
                decoder_hidden[-1], encoder_out, enc_mask
            )

            # Concatenate context with embedded input
            decoder_input = torch.cat([context, embedded[:, t]], dim=-1)
            decoder_input = self.decoder_input_proj(decoder_input).unsqueeze(1)

            # LSTM step
            lstm_out, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )

            # Output
            output_input = torch.cat([context, decoder_hidden[-1]], dim=-1)
            step_output = self.output_proj(output_input)  # (batch, vocab_size)
            outputs.append(step_output.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1) if outputs else torch.empty(
            batch_size, 0, self.vocab_size, device=src.device
        )

        return outputs

    @torch.no_grad()
    def predict(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None,
        sos_idx: int = 1,
        eos_idx: int = 2,
        max_length: int = 100,
    ) -> List[List[int]]:
        '''
        Greedy decoding for inference.

        Args:
            src: (batch, seq_len, input_dim)
            src_lengths: (batch,) input audio lengths
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            max_length: Maximum decode length

        Returns:
            List of decoded token sequences
        '''
        self.eval()
        batch_size = src.size(0)

        # Encode
        encoder_out, encoder_lengths = self.encode(src, src_lengths)

        # Build encoder mask
        enc_mask = torch.arange(
            encoder_out.size(1), device=encoder_out.device
        ).unsqueeze(0) < encoder_lengths.unsqueeze(1)

        # Initial decoder state
        last_indices = (encoder_lengths - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=encoder_out.device)
        last_enc_out = encoder_out[batch_idx, last_indices]
        decoder_hidden = torch.tanh(self.encoder_hidden_proj(last_enc_out))
        decoder_cell = torch.tanh(self.encoder_cell_proj(last_enc_out))
        decoder_hidden = decoder_hidden.unsqueeze(0).expand(
            self.decoder_layers, *decoder_hidden.shape
        ).contiguous()
        decoder_cell = decoder_cell.unsqueeze(0).expand(
            self.decoder_layers, *decoder_cell.shape
        ).contiguous()

        # Greedy decode
        preds = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
        results = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for step in range(max_length):
            # Embed the current prediction
            embedded = self.embedding(preds[:, -1:])  # (batch, 1, hidden_dim)

            # Attention
            context, _ = self._compute_attention(
                decoder_hidden[-1], encoder_out, enc_mask
            )

            # Decoder step
            decoder_input = torch.cat([context, embedded[:, 0]], dim=-1).unsqueeze(1)
            decoder_input = self.decoder_input_proj(decoder_input[:, 0]).unsqueeze(1)
            lstm_out, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )

            # Output
            output_input = torch.cat([context, decoder_hidden[-1]], dim=-1)
            step_logits = self.output_proj(output_input)  # (batch, vocab_size)
            next_token = step_logits.argmax(dim=-1)  # (batch,)

            # Update finished sequences
            for i in range(batch_size):
                if finished[i]:
                    results[i].append(int(eos_idx))
                else:
                    results[i].append(int(next_token[i].item()))
                    if int(next_token[i].item()) == eos_idx:
                        finished[i] = True

            preds = torch.cat([preds, next_token.unsqueeze(1)], dim=1)

            if all(finished):
                break

        return results

    def num_parameters(self) -> int:
        '''Return total number of trainable parameters.'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = Seq2SeqCodeSwitchingTranscriber(vocab_size=52)
    print(f'Seq2Seq Model initialized with {model.num_parameters()} parameters')

    # Quick test
    batch_size, seq_len, input_dim = 2, 1600, 13
    tgt_len = 20
    vocab_size = 52
    src = torch.randn(batch_size, seq_len, input_dim)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    out = model(src, tgt)
    print(f'Output shape: {out.shape}')
    preds = model.predict(src, max_length=15)
    print(f'Predictions: {preds}')
