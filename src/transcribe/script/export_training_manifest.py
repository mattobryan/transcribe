"""Export approved reviewed segments for the existing training command."""

import argparse

from src.transcribe.preprocessing.manifest import export_segments_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export approved segments for training")
    parser.add_argument("--input", required=True, help="Rich review manifest JSON")
    parser.add_argument("--output", required=True, help="Simple metadata JSON")
    args = parser.parse_args()
    count = export_segments_manifest(args.input, args.output)
    print(f"Exported {count} approved training records to {args.output}")


if __name__ == "__main__":
    main()