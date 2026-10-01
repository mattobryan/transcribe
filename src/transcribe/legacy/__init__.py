"""From-scratch MFCC CNN-LSTM models (CTC and Seq2Seq), kept as a learning baseline.

Not the product: see TRD section 2.4 for why. Entry points:
    python -m transcribe.legacy.train --config config/default.yaml
    python -m transcribe.legacy.evaluate --checkpoint checkpoints/final_model.pt
    python -m transcribe.legacy.infer --audio clip.wav --checkpoint checkpoints/final_model.pt
"""
