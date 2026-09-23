"""Review speed from a review manifest's save timestamps.

    python -m src.transcribe.pilot.review_stats data/pilot/NRCCW_KSM09/review_manifest.json

Consecutive saves more than --idle minutes apart start a new session, so
breaks are not counted as review time.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest")
    parser.add_argument("--idle", type=float, default=10.0, help="Minutes that end a session")
    args = parser.parse_args()
    project = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    saved = sorted(
        (datetime.fromisoformat(s["reviewed_at"]), s)
        for s in project.get("segments", []) if s.get("reviewed_at"))
    if not saved:
        print("No reviewed segments yet.")
        return
    work = audio = 0.0
    for i, (when, segment) in enumerate(saved):
        if i:
            gap = (when - saved[i - 1][0]).total_seconds()
            if gap <= args.idle * 60:
                work += gap
        audio += float(segment.get("duration", 0.0))
    approved = sum(s.get("status") == "approved" for _, s in saved)
    print(f"Reviewed segments: {len(saved)} ({approved} approved)")
    print(f"Audio reviewed:    {audio / 60:.1f} min")
    print(f"Review time:       {work / 60:.1f} min (excluding breaks > {args.idle:g} min)")
    if audio:
        print(f"Speed:             {work / audio:.2f} min of review per min of audio")
    edited = sum(1 for _, s in saved if s.get("prefill") is not None and s["transcript"].strip() != s["prefill"].strip())
    print(f"Edited vs prefill: {edited}/{len(saved)} segments")


if __name__ == "__main__":
    main()
