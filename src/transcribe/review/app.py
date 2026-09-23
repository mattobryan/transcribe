"""Small local Gradio editor for candidate audio segments."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from src.transcribe.evaluation.metrics import EvaluationMetrics


def _read(path: str) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run parse_transcript and prepare_audio first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write(path: str, project: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")


def _segments(project: Dict[str, Any]):
    return project.setdefault("segments", [])


def load_project(path: str) -> Tuple[Any, ...]:
    if not path:
        raise ValueError("Manifest path is required")
    project = _read(path)
    segments = _segments(project)
    if not segments:
        raise ValueError("Manifest contains no segments")
    segments = _segments(project)
    saved_index = project.get("current_index")
    if saved_index is None:
        saved_index = next(
            (index for index, segment in enumerate(segments)
             if segment.get("status") != "approved"),
            0,
        )
    return _segment_values(project, path, int(saved_index))


def _segment_values(project: Dict[str, Any], path: str, index: int) -> Tuple[Any, ...]:
    segments = _segments(project)
    index = max(0, min(int(index), len(segments) - 1))
    segment = segments[index]
    audio_file = segment.get("audio_file", "")
    transcript_turns = project.get("transcript_turns", [])
    turn_listing = "\n".join(
        f"{turn.get('turn_id', '')} [{turn.get('speaker_id') or 'unknown'}] {turn.get('text', '')}"
        for turn in transcript_turns
    )
    source_text = ""
    turn_id = segment.get("transcript_turn_id")
    if turn_id:
        source_text = next(
            (turn.get("text", "") for turn in transcript_turns
             if turn.get("turn_id") == turn_id),
            "",
        )
    return (
        project,
        path,
        index + 1,
        len(segments),
        audio_file if Path(audio_file).exists() else None,
        segment.get("transcript", ""),
        segment.get("status", "candidate"),
        float(segment.get("start", 0.0)),
        float(segment.get("end", 0.0)),
        json.dumps(segment.get("vad", {}), ensure_ascii=False, indent=2),
        segment.get("segment_id", str(index)),
        segment.get("transcript_turn_id") or "",
        turn_listing,
        segment.get("asr_hypothesis", ""),
        segment.get("wer"),
        source_text,
        segment.get("speaker_paragraph", False),
        segment.get("unclear", False),
        segment.get("overlap", False),
    )


def select_segment(project: Dict[str, Any], path: str, index: int) -> Tuple[Any, ...]:
    return _segment_values(project, path, index)


def save_segment(
    project: Dict[str, Any], path: str, index: int, transcript: str,
    status: str, start: float, end: float, transcript_turn_id: str,
    asr_hypothesis: str = "", source_text: str = "",
    speaker_paragraph: bool = False, unclear: bool = False,
    overlap: bool = False,
) -> Tuple[Any, ...]:
    segments = _segments(project)
    index = max(0, min(int(index) - 1, len(segments) - 1))
    segment = segments[index]
    segment.update({
        "transcript": transcript,
        "status": status,
        "start": float(start),
        "end": float(end),
        "transcript_turn_id": transcript_turn_id or None,
        "asr_hypothesis": asr_hypothesis,
        "speaker_paragraph": bool(speaker_paragraph),
        "unclear": bool(unclear),
        "overlap": bool(overlap),
        "review": {"status": status},
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    project["current_index"] = index
    hypothesis = asr_hypothesis.strip()
    if hypothesis and transcript.strip():
        metrics = EvaluationMetrics(char_to_idx={}, idx_to_char={})
        segment["wer"] = metrics.calculate_wer(transcript, hypothesis)
    _write(path, project)
    return _segment_values(project, path, index)


def save_and_next(
    project: Dict[str, Any], path: str, index: int, transcript: str,
    start: float, end: float, transcript_turn_id: str, asr_hypothesis: str,
    speaker_paragraph: bool, unclear: bool, overlap: bool,
) -> Tuple[Any, ...]:
    """Approve the current segment, save it, and load the next segment."""
    current = save_segment(
        project, path, index, transcript, "approved",
        start, end, transcript_turn_id, asr_hypothesis, "",
        speaker_paragraph, unclear, overlap,
    )
    next_index = min(int(index) + 1, int(current[3]))
    current[0]["current_index"] = next_index - 1
    _write(current[1], current[0])
    return select_segment(current[0], current[1], next_index - 1)


def _model_transcribe(audio_file: str, checkpoint_path: str) -> str:
    """Run the repository model on one chunk when a checkpoint is supplied."""
    import pickle
    import torch
    from src.transcribe.config import load_config
    from src.transcribe.data.preprocessing import DataPreprocessor
    from src.transcribe.models.ctc_model import CTCCodeSwitchingTranscriber
    from src.transcribe.models.seq2seq import Seq2SeqCodeSwitchingTranscriber

    # Checkpoints are saved locally by this repo; disable weights_only to allow
    # the stored config dict (which may contain pathlib objects) to unpickle.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_dir = Path(checkpoint_path).parent
    with (checkpoint_dir / "vocab.pkl").open("rb") as handle:
        vocab_data = pickle.load(handle)
    saved = checkpoint.get("config", {})
    model_type = checkpoint.get("model_type", saved.get("model_type", "ctc"))
    model_args = {
        "input_dim": saved.get("n_mfcc", 13),
        "hidden_dim": saved.get("hidden_dim", 256),
        "num_layers": saved.get("num_layers", 3),
        "vocab_size": len(vocab_data["vocab"]),
        "dropout": saved.get("dropout", 0.2),
        "bidirectional": saved.get("bidirectional", True),
    }
    if model_type == "ctc":
        model = CTCCodeSwitchingTranscriber(**model_args)
    else:
        model = Seq2SeqCodeSwitchingTranscriber(
            **model_args, decoder_layers=saved.get("decoder_layers", 2),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = DataPreprocessor(
        sample_rate=saved.get("sample_rate", 16000),
        n_mfcc=saved.get("n_mfcc", 13),
        n_fft=saved.get("n_fft", 2048),
        hop_length=saved.get("hop_length", 512),
        max_audio_seconds=saved.get("max_audio_length"),
    )
    features = torch.FloatTensor(preprocessor.load_audio(audio_file)).unsqueeze(0)
    lengths = torch.tensor([features.size(1)], dtype=torch.long)
    idx_to_char = {int(key): value for key, value in vocab_data["idx_to_char"].items()}
    if model_type == "ctc":
        tokens = model.predict(features, lengths, blank_idx=0)[0]
        return "".join(idx_to_char.get(int(token), "") for token in tokens)
    sos = vocab_data["char_to_idx"]["<SOS>"]
    eos = vocab_data["char_to_idx"]["<EOS>"]
    tokens = model.predict(features, lengths, sos_idx=sos, eos_idx=eos)[0]
    return "".join(
        idx_to_char.get(int(token), "") for token in tokens if int(token) != eos
    )


def transcribe_current(project: Dict[str, Any], path: str, index: int, checkpoint: str):
    if not checkpoint:
        return (*_segment_values(project, path, max(0, int(index) - 1)),
                "No checkpoint supplied. Type a .pt path in the 'Checkpoint (.pt)' box below "
                "(e.g. checkpoints/final_model.pt), or use 'Save and Continue' to approve "
                "manually until a model is trained.")
    if not Path(checkpoint).exists():
        return (*_segment_values(project, path, max(0, int(index) - 1)),
                f"Checkpoint not found: {checkpoint}. Verify the path, or train one with "
                "`python -m src.transcribe.script.train --config config/default.yaml`.")
    internal_index = max(0, min(int(index) - 1, len(_segments(project)) - 1))
    segment = _segments(project)[internal_index]
    try:
        hypothesis = _model_transcribe(segment["audio_file"], checkpoint)
    except Exception as exc:
        return (*_segment_values(project, path, internal_index),
                f"Transcription failed: {exc}")
    # The hypothesis is stored beside the reviewer's text and never replaces it.
    segment["asr_hypothesis"] = hypothesis
    project["current_index"] = internal_index
    _write(path, project)
    return (*_segment_values(project, path, internal_index), "Transcription completed.")


def finish_audio_file(project: Dict[str, Any], path: str):
    segments = _segments(project)
    approved = sum(segment.get("status") == "approved" for segment in segments)
    project["finished"] = approved == len(segments)
    project["finished_segments"] = approved
    project["total_segments"] = len(segments)
    _write(path, project)
    return f"Finished: {approved}/{len(segments)} chunks approved. " \
        f"Manifest saved to {path}"


def save_and_exit(
    project: Dict[str, Any], path: str, index: int, transcript: str,
    status: str, start: float, end: float, transcript_turn_id: str,
    asr_hypothesis: str, speaker_paragraph: bool, unclear: bool,
    overlap: bool,
):
    save_segment(
        project, path, index, transcript, status, start, end,
        transcript_turn_id, asr_hypothesis, "", speaker_paragraph,
        unclear, overlap,
    )
    project["review_session"] = "paused"
    _write(path, project)
    return "Saved. You can close this tab and resume later."


def launch(manifest_path: str, checkpoint_path: str = "", share: bool = False) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Install annotation dependencies with: python -m pip install -r requirements-annotation.txt"
        ) from exc

    initial = load_project(manifest_path)
    if not checkpoint_path:
        root = Path(manifest_path).resolve().parents[2]
        for candidate in (root / "checkpoints" / "final_model.pt", root / "checkpoints" / "best_model.pt"):
            if candidate.exists():
                checkpoint_path = str(candidate)
                break
    earth_css = """
    :root {
        --review-bg: #f4eee5;
        --review-surface: #fffaf3;
        --review-text: #2d241d;
        --review-muted: #675443;
        --review-border: #b99b7a;
        --review-accent: #8a5a3b;
        --review-accent-hover: #6f432d;
    }
    .dark {
        --review-bg: #211b17;
        --review-surface: #30261f;
        --review-text: #f4eee5;
        --review-muted: #d5bda5;
        --review-border: #806149;
        --review-accent: #b77b51;
        --review-accent-hover: #d19669;
    }
    @media (prefers-color-scheme: dark) {
        :root:not(.light) {
            --review-bg: #211b17;
            --review-surface: #30261f;
            --review-text: #f4eee5;
            --review-muted: #d5bda5;
            --review-border: #806149;
            --review-accent: #b77b51;
            --review-accent-hover: #d19669;
        }
    }
    body, .gradio-container {
        background: var(--review-bg) !important;
        color: var(--review-text) !important;
        font-family: 'Times New Roman', serif !important;
        font-size: 12pt !important;
    }
    .gradio-container *, .gradio-container button, .gradio-container input,
    .gradio-container textarea, .gradio-container label {
        font-family: 'Times New Roman', serif !important;
        font-size: 12pt !important;
    }
    .gradio-container input, .gradio-container textarea,
    .gradio-container .wrap, .gradio-container .block,
    .gradio-container .panel {
        background: var(--review-surface) !important;
        color: var(--review-text) !important;
        border-color: var(--review-border) !important;
    }
    .gradio-container label, .gradio-container .label-wrap span,
    .gradio-container h1, .gradio-container h2, .gradio-container h3,
    .gradio-container p { color: var(--review-text) !important; }
    .gradio-container button {
        background: var(--review-accent) !important;
        color: #fffaf3 !important;
        border-color: var(--review-accent) !important;
    }
    .gradio-container button:hover { background: var(--review-accent-hover) !important; }
    """
    with gr.Blocks(title="Chunk Transcription Review", css=earth_css) as demo:
        project_state = gr.State(initial[0])
        path_state = gr.State(initial[1])
        gr.Markdown("# Chunk Transcription Review")
        with gr.Row():
            index = gr.Number(label="Chunk", value=initial[2], precision=0)
            count = gr.Number(label="Total chunks", value=initial[3], interactive=False)
        with gr.Row():
            segment_id = gr.Textbox(label="Chunk ID", value=initial[10], interactive=False)
            transcribe_status = gr.Textbox(label="Transcription status", interactive=False)
        audio = gr.Audio(label="Audio chunk", value=initial[4], type="filepath")
        transcribe = gr.Button("Transcribe", variant="primary")
        checkpoint_box = gr.Textbox(
            label="Checkpoint (.pt)", value=checkpoint_path or "",
            placeholder="checkpoints/final_model.pt",
        )
        transcript = gr.Textbox(
            label="Edit transcript", value=initial[5], lines=7,
        )
        with gr.Row():
            previous = gr.Button("Previous chunk")
            next_button = gr.Button("Next chunk")
            save_continue = gr.Button("Save and Continue", variant="primary")
            save_exit = gr.Button("Save and Exit")
        hypothesis = gr.Textbox(
            label="Automatic model transcript (recorded)", value=initial[13],
            lines=2, interactive=False, visible=False,
        )
        source_text = gr.Textbox(
            label="Full reviewed script", value=initial[12],
            lines=12, interactive=False,
        )
        # Keep annotation metadata in the manifest without putting it in the primary UI.
        status = gr.State(initial[6])
        start = gr.State(initial[7])
        end = gr.State(initial[8])
        vad = gr.State(initial[9])
        transcript_turn_id = gr.State(initial[11])
        transcript_turns = gr.State(initial[12])
        wer = gr.State(initial[14])
        speaker_paragraph = gr.State(initial[16])
        unclear = gr.State(initial[17])
        overlap = gr.State(initial[18])
        review_status = gr.Textbox(label="Status", interactive=False)
        with gr.Accordion("Chunk metadata", open=False):
            gr.Code(
                label="Technical metadata", language="json",
                value=json.dumps({
                    "segment_id": initial[10], "start": initial[7],
                    "end": initial[8], "wer": initial[14],
                    "checkpoint": checkpoint_path or None,
                }, indent=2),
            )

        fields = [project_state, path_state, index, count, audio, transcript, status, start, end, vad, segment_id, transcript_turn_id, transcript_turns, hypothesis, wer, source_text, speaker_paragraph, unclear, overlap]

        def select_display(p, path, displayed_index):
            return _segment_values(p, path, max(0, int(displayed_index) - 1))

        previous.click(lambda p, path, i: _segment_values(p, path, max(0, int(i) - 2)), [project_state, path_state, index], fields)
        next_button.click(lambda p, path, i: _segment_values(p, path, min(len(_segments(p)) - 1, int(i))), [project_state, path_state, index], fields)
        index.change(select_display, [project_state, path_state, index], fields)
        transcribe.click(
            transcribe_current,
            [project_state, path_state, index, checkpoint_box],
            fields + [transcribe_status],
        )
        save_continue.click(save_and_next, [project_state, path_state, index, transcript, start, end, transcript_turn_id, hypothesis, speaker_paragraph, unclear, overlap], fields)
        save_exit.click(save_and_exit, [project_state, path_state, index, transcript, status, start, end, transcript_turn_id, hypothesis, speaker_paragraph, unclear, overlap], review_status)
    demo.launch(share=share)