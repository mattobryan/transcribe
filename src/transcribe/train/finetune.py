"""Fine-tune Whisper on the corrected chunks (LoRA by default), scoring the dev set after every epoch.

    python -m transcribe.train.finetune data/train --model openai/whisper-small --out checkpoints/run1 --epochs 3

Needs ``data/train/train.jsonl`` and ``dev.jsonl`` from ``transcribe.train.prepare``. Each chunk is trained with the language token
of its main language (Swahili if it has more Swahili than English words), and run at test time with the same rule, 'auto', or a
fixed language, so you can see which works best for code-switched speech.
Saves the LoRA adapter in ``<out>/adapter`` (or the whole model with --full). The best dev WER is kept.
"""

import argparse
import json
import random
import time

import numpy as np
from pathlib import Path

from .dataset import read_manifest
from .evaluate import score_system, transcribe_rows


class Clips:
    def __init__(self, rows, processor):
        self.rows, self.processor = rows, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import soundfile as sf

        row = self.rows[i]
        audio = sf.read(row["audio"], dtype="float32")[0]
        feats = self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]
        tok = self.processor.tokenizer
        tok.set_prefix_tokens(language=row["language"], task="transcribe")
        return {"input_features": feats, "labels": tok(row["text"]).input_ids[:440]}


def collate(processor, start_id):
    import torch

    def run(batch):
        feats = torch.from_numpy(np.stack([b["input_features"] for b in batch]))
        longest = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for k, b in enumerate(batch):
            labels[k, :len(b["labels"])] = torch.tensor(b["labels"])
        if (labels[:, 0] == start_id).all():             # the model adds the start token itself
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}
    return run


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", help="folder with train.jsonl and dev.jsonl")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--out", default="checkpoints/run1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="1e-4 for LoRA, 1e-5 for --full")
    parser.add_argument("--full", action="store_true", help="train every weight instead of a LoRA adapter")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (for a quick test)")
    parser.add_argument("--eval-language", default="dominant")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    train_rows, dev_rows = read_manifest(data / "train.jsonl"), read_manifest(data / "dev.jsonl")
    if not train_rows:
        raise SystemExit("train.jsonl is empty")
    print(f"{len(train_rows)} train chunks ({sum(r['duration'] for r in train_rows) / 3600:.2f} h), {len(dev_rows)} dev chunks, device {device}", flush=True)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    if not args.full:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"))
        model.print_trainable_parameters()
    model.to(device)

    start_id = model.config.decoder_start_token_id if hasattr(model, "config") else processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    loader = DataLoader(Clips(train_rows, processor), batch_size=args.batch, shuffle=True, collate_fn=collate(processor, start_id), num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total = args.max_steps or args.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total // 20), total)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, step, history = None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.time(), []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running.append(float(loss))
            step += 1
            if step % 20 == 0:
                print(f"  step {step}/{total} loss {sum(running[-20:]) / len(running[-20:]):.3f}", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        entry = {"epoch": epoch, "train_loss": round(sum(running) / max(1, len(running)), 4), "minutes": round((time.time() - started) / 60, 1)}
        if dev_rows:
            metrics = score_system(dev_rows, transcribe_rows(model, processor, dev_rows, args.eval_language, args.batch, device))
            entry.update(dev_wer=metrics["wer"], dev_wer_by_lang=metrics["wer_by_lang"], dev_switch_point_error_rate=metrics["switch_point_error_rate"])
        history.append(entry)
        print(entry, flush=True)
        score = entry.get("dev_wer")
        if best is None or (score is not None and score < best):
            best = score
            model.save_pretrained(out / ("model" if args.full else "adapter"))
            processor.save_pretrained(out)
        if args.max_steps and step >= args.max_steps:
            break
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print("best dev WER", best, "->", out)


if __name__ == "__main__":
    main()
