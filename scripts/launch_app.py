"""启动本项目的本地服务，等待就绪后再打开浏览器。"""
import argparse
import importlib.util
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def check_environment():
    missing = [name for name in ("streamlit", "torch", "torchvision", "PIL")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing) + ". Run setup.bat first.")
    if not (ROOT / "outputs" / "baseline" / "best.pt").is_file():
        raise RuntimeError("Model missing: outputs/baseline/best.pt. Copy the trained model or train first.")


def available_port(start=8501, attempts=20):
    for port in range(start, min(start + attempts, 65536)):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free local port found. Close an old demo window and try again.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    check_environment()
    if args.check:
        print("Ready: environment and trained model found.")
        return 0
    port = available_port()
    url = f"http://127.0.0.1:{port}"
    command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
               "--server.address", "127.0.0.1", "--server.port", str(port),
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    process = subprocess.Popen(command, cwd=str(ROOT))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        ready = False
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before startup. See the error above.")
            try:
                with opener.open(url + "/_stcore/health", timeout=0.5) as response:
                    ready = response.read().strip() == b"ok"
            except (OSError, urllib.error.URLError):
                pass
            if ready:
                break
            time.sleep(0.25)
        if not ready:
            raise RuntimeError("Startup timed out. See the Streamlit output above.")
        print(f"\nDemo ready: {url}\nKeep this terminal open. Press Ctrl+C to stop.\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as error:
        print(f"\n{error}", file=sys.stderr)
        sys.exit(1)
