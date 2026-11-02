"""Drive the chat UI with headless Chromium and save docs/chat.png.

Usage:  uv run python scripts/screenshot.py
Runs the server with the mock provider on a throwaway database, so no API key is needed.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.getenv("SCREENSHOT_PORT", "8765"))
CONVERSATION = [
    "How long does shipping to Canada take?",
    "Where is my order NW-10482?",
    "The zipper on the jacket broke on day two. Can I get it replaced?",
    "ana.ruiz@example.com",
]


def wait_for_server(url: str, timeout: float = 20) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server did not start at {url}")


def chromium_executable() -> str | None:
    # Use an already-downloaded Chromium if the bundled revision isn't installed.
    found = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))
    return found[-1] if found else None


def main() -> None:
    tmp = tempfile.mkdtemp()
    env = {**os.environ, "LLM_PROVIDER": "mock", "EMBEDDINGS": "none", "DB_PATH": f"{tmp}/demo.db"}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "support_agent.app:create_app", "--factory", "--port", str(PORT)],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{PORT}"
        wait_for_server(f"{base}/health")
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception:
                browser = p.chromium.launch(executable_path=chromium_executable())
            page = browser.new_page(viewport={"width": 1360, "height": 1480}, device_scale_factor=2)
            page.goto(base)
            for i, message in enumerate(CONVERSATION, start=1):
                page.fill("#input", message)
                page.click("#send")
                page.wait_for_function(
                    "n => document.querySelectorAll('.msg.bot .bubble:not(.typing)').length >= n", arg=i
                )
            page.wait_for_selector(".ticket")
            page.wait_for_timeout(300)
            out = ROOT / "docs" / "chat.png"
            out.parent.mkdir(exist_ok=True)
            page.screenshot(path=str(out))
            browser.close()
            print(f"saved {out}")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
