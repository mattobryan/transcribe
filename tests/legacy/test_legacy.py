"""Legacy baseline: grouped split (P0.4) and a toy training run (P0.2)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from transcribe.legacy.data.preprocessing import DataPreprocessor, recording_of

SR = 16000


def make_corpus(root: Path, recordings: int, chunks: int) -> Path:
    rng = np.random.default_rng(0)
    items = []
    for r in range(recordings):
        for c in range(chunks):
            path = root / f"REC{r:02d}_{c:06d}.wav"
            sf.write(path, (rng.standard_normal(SR * 2) * 0.05).astype(np.float32), SR)
            items.append({"audio_file": str(path), "transcript": f"habari {chr(97 + r)}{c} yako"})
    metadata = root / "metadata.json"
    metadata.write_text(json.dumps(items))
    return metadata


def test_recording_of():
    assert recording_of("data/chunks/NRCCW_KSM09_000123.wav") == "NRCCW_KSM09"
    assert recording_of("sample1.wav") == "sample1"


def test_split_never_shares_a_recording(tmp_path):
    metadata = make_corpus(tmp_path, recordings=10, chunks=4)
    data = DataPreprocessor().prepare_dataset(str(tmp_path), str(metadata))
    groups = data["groups"]
    assert not set(groups["train"]) & set(groups["test"])
    assert not set(groups["train"]) & set(groups["val"])
    assert not set(groups["val"]) & set(groups["test"])
    assert groups["test"] and groups["val"] and groups["train"]
    for split in ("train", "val", "test"):
        for path in data[split]["audio"]:
            assert recording_of(path) in groups[split]


def test_vocabulary_comes_from_train_only(tmp_path):
    metadata = make_corpus(tmp_path, recordings=10, chunks=4)
    data = DataPreprocessor().prepare_dataset(str(tmp_path), str(metadata))
    train_letters = {ch for path in data["train"]["audio"]
                     for item in json.loads(metadata.read_text()) if item["audio_file"] == path
                     for ch in item["transcript"] if ch != " "}
    assert set(data["vocab"][4:]) == train_letters


def test_few_recordings_split_by_position(tmp_path):
    metadata = make_corpus(tmp_path, recordings=1, chunks=10)
    with pytest.warns(UserWarning):
        data = DataPreprocessor().prepare_dataset(str(tmp_path), str(metadata))
    assert [Path(p).stem[-2:] for p in data["test"]["audio"]] == ["08", "09"]


def test_toy_training_run(tmp_path):
    pytest.importorskip("torch")
    metadata = make_corpus(tmp_path, recordings=4, chunks=3)
    config = tmp_path / "config.yaml"
    config.write_text("\n".join([
        f'data_dir: "{tmp_path}"', f'metadata_file: "{metadata}"', f'checkpoint_dir: "{tmp_path / "ckpt"}"',
        "model_type: ctc", "hidden_dim: 16", "num_layers: 1", "batch_size: 2", "epochs: 1",
        "save_every: 1", "eval_every: 1", "device: cpu",
    ]))
    result = subprocess.run([sys.executable, "-m", "transcribe.legacy.train", "--config", str(config)],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (tmp_path / "ckpt" / "vocab.pkl").exists()
    assert list((tmp_path / "ckpt").glob("*.pt"))
