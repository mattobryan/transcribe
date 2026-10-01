"""Start the transcription app.

    python -m transcribe.app                     # http://127.0.0.1:8000
    python -m transcribe.app --model large-v3-turbo --port 8080
    python -m transcribe.app --open data/pilot/NRCCW_KSM09/review_manifest.json
"""

import argparse
import threading
import webbrowser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcription app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", default="data/app", help="Where sessions and uploads are kept")
    parser.add_argument("--open", dest="manifest", help="Open a pilot review_manifest.json as a session")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    import uvicorn

    from .server import create_app, import_manifest

    app = create_app(Path(args.data))
    url = f"http://{args.host}:{args.port}/"
    if args.manifest:
        session_id = import_manifest(app.state.store, Path(args.manifest))
        url += f"#/edit/{session_id}"
        print(f"Opened {args.manifest} as session {session_id}")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Transcription app: {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
