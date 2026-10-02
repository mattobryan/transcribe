"""Run the app on a cloud GPU (Colab or Kaggle) and open it from your own browser.

    python -m transcribe.app.online              # public link + token (Kaggle, or anywhere)
    python -m transcribe.app.online --no-tunnel  # app only, port 8000 (Colab opens it itself)

Starts the app, creates a temporary public https link with a Cloudflare quick tunnel and
prints it with a private token. Only people who have the full link (with ?token=...) can
open it. Stop the cell to take the link down; the notebook's disk is wiped when the session ends,
so download your transcripts (TXT / SRT / JSON, top right) before you close it.
"""

import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
PORT = 8000


def nvidia_lib_dirs():
    """cuBLAS / cuDNN pip wheels (installed with torch on Colab and Kaggle) that faster-whisper needs."""
    import importlib.util
    dirs = []
    for name in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec and spec.submodule_search_locations:
            lib = Path(list(spec.submodule_search_locations)[0]) / "lib"
            if lib.is_dir():
                dirs.append(str(lib))
    return dirs


def reexec_with_gpu_libs() -> None:
    """LD_LIBRARY_PATH is read when a process starts, so set it and start again once."""
    if os.environ.get("TRANSCRIBE_REEXEC"):
        return
    dirs = nvidia_lib_dirs()
    if dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["TRANSCRIBE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, "-u", "-m", "transcribe.app.online", *sys.argv[1:]])


def get_cloudflared() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    target = Path("/tmp/cloudflared")
    if not target.exists():
        print("Downloading the tunnel program (one time, about 40 MB) ...", flush=True)
        urllib.request.urlretrieve(CLOUDFLARED, target)
        target.chmod(target.stat().st_mode | stat.S_IEXEC)
    return str(target)


def open_tunnel(port: int, timeout: float = 60.0):
    process = subprocess.Popen([get_cloudflared(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found = {}
    tail = []

    def read():
        for line in process.stdout:
            tail.append(line.rstrip())
            del tail[:-8]
            match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
            if match and "url" not in found:
                found["url"] = match.group()
    threading.Thread(target=read, daemon=True).start()
    deadline = time.time() + timeout
    while time.time() < deadline and "url" not in found and process.poll() is None:
        time.sleep(0.5)
    if "url" not in found:
        process.terminate()
        raise RuntimeError("Could not create the public link (is Internet turned on for this notebook?)\n"
                           + "\n".join(tail))
    return process, found["url"]


def main() -> None:
    reexec_with_gpu_libs()
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-tunnel", action="store_true",
                        help="Only start the app on port 8000 (Colab opens it with serve_kernel_port_as_window)")
    args = parser.parse_args()

    from .server import create_app

    token = None if args.no_tunnel else (os.environ.get("TRANSCRIBE_TOKEN") or secrets.token_urlsafe(12))
    data = Path(os.environ.get("TRANSCRIBE_DATA", "/tmp/transcribe-app"))
    try:
        from transcribe.pilot.asr import cuda_available
        print("GPU:", "yes" if cuda_available() else "NO (transcribing will be slow; turn on a GPU in the notebook settings)",
              flush=True)
    except Exception:
        pass
    app = create_app(data, token=token)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.2)
    if args.no_tunnel:
        print(f"App running on port {PORT}", flush=True)
        while True:
            time.sleep(60)
    process, url = open_tunnel(PORT)
    print("\n" + "=" * 70)
    print("OPEN THIS LINK IN YOUR BROWSER (keep it private):\n")
    print(f"    {url}/?token={token}\n")
    print("Keep this cell running. Stop it to take the link down. Download your")
    print("transcripts (TXT / SRT / JSON, top right) before you close the notebook.")
    print("=" * 70 + "\n", flush=True)
    try:
        while process.poll() is None:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()


if __name__ == "__main__":
    main()
