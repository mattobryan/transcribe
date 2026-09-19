"""Launch the local speech alignment review UI."""

import argparse

from src.transcribe.review.app import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="Review speech alignment manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    launch(args.manifest, checkpoint_path=args.checkpoint, share=args.share)


if __name__ == "__main__":
    main()