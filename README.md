# English-Swahili Code-Switching Transcriber

A custom speech-to-text transcriber for English-Swahili code-switching speech patterns commonly found in Kenya.

## Features

- Handles mixed English-Swahili speech recognition
- Two model architectures:
  - **CTC** (recommended): CNN-LSTM with CTC loss (alignment-free training)
  - **Seq2Seq**: Encoder-decoder with attention and teacher forcing
- CNN-LSTM backbone with BatchNorm
- Support for code-switched vocabulary (English + Swahili characters)
- Evaluation metrics: WER, CER, and BLEU scores

## Project Structure

```
transcribe/
|-- config/
|   `-- default.yaml             # Configuration file
|-- src/transcribe/
|   |-- __init__.py              # Package root
|   |-- config.py                # Config dataclass + YAML loader
|   |-- models/
|   |   |-- __init__.py
|   |   |-- ctc_model.py         # CNN-LSTM + CTC
|   |   `-- seq2seq.py           # Encoder-decoder + attention
|   |-- data/
|   |   |-- __init__.py
|   |   |-- preprocessing.py     # Audio + vocab utilities
|   |   `-- dataset.py           # PyTorch Dataset + collate_fn
|   |-- evaluation/
|   |   |-- __init__.py
|   |   `-- metrics.py           # WER, CER, BLEU
|   |-- training/
|   |   |-- __init__.py
|   |   `-- trainer.py           # Training loop + checkpointing
|   `-- script/
|       |-- __init__.py
|       |-- train.py             # Training entry point
|       |-- evaluate.py          # Evaluation entry point
|       `-- infer.py             # Inference entry point
|-- data/                        # Audio data + metadata (gitignored)
|-- checkpoints/                 # Model checkpoints (gitignored)
|-- requirements.txt
|-- pyproject.toml
|-- README.md
`-- .gitignore
```

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

> **Use `python -m pip`, not a bare `pip`.** On this (and many Windows) systems,
> a bare `pip` may point to a *different* Python instalment (e.g. 3.13) than the
> `python` that runs this project (3.10), so packages you install with `pip`
> won't be visible to `python`. `python -m pip` always targets the interpreter
> you're using. If `pip` isn't available on `python`, run `python -m ensurepip`.

## Quick Start

### 1. Prepare Your Data

Place audio files (`.wav`) in `data/audio/` and create `data/metadata.json`:

```json
[
  {"audio_file": "sample1.wav", "transcript": "habari yako jambo"},
  {"audio_file": "sample2.wav", "transcript": "how are you mahaba"}
]
```

### 2. Train

```bash
python -m src.transcribe.script.train --config config/default.yaml
```

### 3. Evaluate

```bash
python -m src.transcribe.script.evaluate --checkpoint checkpoints/final_model.pt
```

### 4. Infer (Transcribe New Audio)

```bash
python -m src.transcribe.script.infer --audio data/audio/sample1.wav --checkpoint checkpoints/final_model.pt
```

## Configuration

Settings are in `config/default.yaml`. Key fields:

| Field             | Description                         | Default |
|-------------------|-------------------------------------|---------|
| `sample_rate`     | Audio sampling rate (Hz)            | 16000   |
| `n_mfcc`          | Number of MFCC coefficients         | 13      |
| `model_type`      | `ctc` (recommended) or `seq2seq`    | ctc     |
| `hidden_dim`      | LSTM hidden units                   | 256     |
| `num_layers`      | LSTM layers                         | 3       |
| `bidirectional`   | Bidirectional LSTM                  | true    |
| `learning_rate`   | Initial LR                          | 0.001   |
| `batch_size`      | Training batch size                 | 16      |
| `epochs`          | Number of training epochs           | 100     |
| `eval_every`      | Run validation every N epochs       | 5       |
| `save_every`      | Save checkpoint every N epochs      | 10      |
| `grad_clip`       | Max gradient norm                   | 1.0     |

## Model Architectures

### CTC Model (Recommended)

- CNN feature extractor (3 Conv1D layers, reduces sequence by 8x)
- Bidirectional LSTM encoder
- Per-frame logits with CTC loss
- Greedy decoding or beam search

### Seq2Seq Model

- CNN + BiLSTM encoder
- LSTM decoder with additive attention
- Teacher forcing during training (configurable ratio)
- Greedy decoding during inference

## Dependencies

- `torch` — deep learning (CPU build fine)
- `librosa`, `soundfile`, `numba` — audio loading + MFCC features
- `numpy`, `scipy`, `scikit-learn` — scientific stack / train-test splits
- `editdistance`, `python-Levenshtein` — WER/CER edit distance
- `nltk` — BLEU score
- `PyYAML` — config loading