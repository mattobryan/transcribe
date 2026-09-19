"""
Configuration management for the transcriber.
Loads settings from YAML config file with defaults.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class Config:
    """Configuration dataclass with all settings."""
    
    # Data settings
    data_dir: str = "./data/audio"
    metadata_file: str = "./data/metadata.json"
    checkpoint_dir: str = "./checkpoints"
    
    # Audio processing
    sample_rate: int = 16000
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    max_audio_length: Optional[int] = None  # seconds
    
    # Model architecture
    model_type: str = "ctc"  # "ctc" or "seq2seq"
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    bidirectional: bool = True
    
    # Seq2Seq specific
    decoder_layers: int = 2
    teacher_forcing_ratio: float = 0.5
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 100
    save_every: int = 10
    eval_every: int = 5
    grad_clip: float = 1.0
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Language settings
    languages: list = field(default_factory=lambda: ["english", "swahili"])
    code_switching: bool = True
    
    # Vocabulary settings
    min_char_frequency: int = 5
    max_sequence_length: int = 200
    
    # Evaluation
    metrics: list = field(default_factory=lambda: ["wer", "cer", "bleu"])
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    def __post_init__(self):
        """Resolve paths and validate settings."""
        # Convert to Path objects for easier handling
        self.data_dir = Path(self.data_dir)
        self.metadata_file = Path(self.metadata_file)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        
        # Auto-detect device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with defaults.
    
    Args:
        config_path: Path to YAML config file. If None, uses default config.
        
    Returns:
        Config object with all settings.
    """
    # Default config values
    defaults = {
        "data_dir": "./data/audio",
        "metadata_file": "./data/metadata.json",
        "checkpoint_dir": "./checkpoints",
        "sample_rate": 16000,
        "n_mfcc": 13,
        "n_fft": 2048,
        "hop_length": 512,
        "max_audio_length": None,
        "model_type": "ctc",
        "hidden_dim": 256,
        "num_layers": 3,
        "dropout": 0.2,
        "bidirectional": True,
        "decoder_layers": 2,
        "teacher_forcing_ratio": 0.5,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 100,
        "save_every": 10,
        "eval_every": 5,
        "grad_clip": 1.0,
        "scheduler_patience": 5,
        "scheduler_factor": 0.5,
        "languages": ["english", "swahili"],
        "code_switching": True,
        "min_char_frequency": 5,
        "max_sequence_length": 200,
        "metrics": ["wer", "cer", "bleu"],
        "device": "auto",
    }
    
    # Load from YAML if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
        defaults.update(yaml_config)
    
    return Config(**defaults)


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = {
        k: str(v) if isinstance(v, Path) else v 
        for k, v in config.__dict__.items()
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)