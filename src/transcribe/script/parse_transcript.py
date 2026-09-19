"""Convert a reviewed DOCX transcript into a preprocessing manifest."""

import argparse

from src.transcribe.preprocessing.docx_parser import parse_docx
from src.transcribe.preprocessing.manifest import save_manifest
from src.transcribe.preprocessing.pdf_parser import parse_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a reviewed transcript DOCX or PDF")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--recording-id")
    parser.add_argument("--format", choices=["auto", "docx", "pdf"], default="auto")
    args = parser.parse_args()

    parser_fn = parse_docx
    if args.format == "pdf" or (args.format == "auto" and args.transcript.lower().endswith(".pdf")):
        parser_fn = parse_pdf
    manifest = parser_fn(
        args.transcript,
        recording_id=args.recording_id,
        source_audio=args.audio,
    )
    save_manifest(manifest, args.output)
    print(f"Parsed {len(manifest.transcript_turns)} transcript turns")
    print(f"Saved manifest to {args.output}")


if __name__ == "__main__":
    main()