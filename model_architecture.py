import torch
import torch.nn as nn
import torch.nn.functional as F

class CodeSwitchingTranscriber(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=256, num_layers=3, 
                 vocab_size=50, dropout=0.2, bidirectional=True):
        super(CodeSwitchingTranscriber, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        
        # CNN feature extractor for audio preprocessing
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate CNN output dimension
        cnn_out_dim = 256
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
        
        # Output layer
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(output_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        
        # CNN feature extraction
        # Transpose for CNN: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # Transpose back: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final output
        output = self.dropout(context)
        output = self.fc(output)
        
        return output
    
    def predict_sequence(self, x, max_length=100):
        """Generate prediction sequence"""
        with torch.no_grad():
            # CNN feature extraction
            cnn_features = self.cnn(x.transpose(1, 2)).transpose(1, 2)
            
            # LSTM processing
            lstm_out, _ = self.lstm(cnn_features)
            
            # Apply attention
            attention_weights = F.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)
            
            # Final output
            output = self.dropout(context)
            output = self.fc(output)
            
            return output

class CodeSwitchingTranscriberSeq2Seq(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=256, num_layers=2,
                 vocab_size=50, dropout=0.2, bidirectional=True):
        super(CodeSwitchingTranscriberSeq2Seq, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # src: (batch, seq_len, input_dim)
        # tgt: (batch, seq_len) - target indices
        
        # Encode
        encoder_out, (hidden, cell) = self.encoder_lstm(src)
        
        # Decode
        embedded = self.dropout(self.embedding(tgt))
        
        # Simple decoder without attention for now
        decoder_out, _ = self.decoder_lstm(embedded)
        
        # Output
        output = self.fc(decoder_out)
        
        return output

if __name__ == "__main__":
    model = CodeSwitchingTranscriber()
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
