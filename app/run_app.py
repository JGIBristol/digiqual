import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


# --- Logging: write to a fixed, discoverable file so a crashed --windowed
# build (no console) can still be diagnosed after the fact. ---
def _resolve_log_path() -> Path:
    """Picks a per-user, writable location for the app's log file."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path.home() / ".local" / "share"
    log_dir = base / "Digiqual"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "digiqual.log"


logging.basicConfig(
    filename=_resolve_log_path(),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# --- 0. Environment Fixes (Windows PyInstaller, Proxies, pythonnet) ---
# Ensure local connections bypass any corporate/university proxy servers
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

# Python.Runtime.dll requires an explicit pointer to python311.dll in frozen bundles
if sys.platform == "win32" and getattr(sys, "frozen", False):
    bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    py_dlls = list(bundle_dir.glob("python3*.dll")) or list(
        (bundle_dir / "_internal").glob("python3*.dll")
    )
    if py_dlls:
        os.environ["PYTHONNET_PYDLL"] = str(py_dlls[0].resolve())

from shiny import run_app

from app import app

logger = logging.getLogger(__name__)

# --- 1. Set Working Directory ---
if getattr(sys, "frozen", False):
    os.chdir(sys._MEIPASS)
    sys.path.insert(0, sys._MEIPASS)


# --- 2. Port Helper ---
def get_free_port() -> int:
    """Finds an available local TCP port dynamically."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


SELECTED_PORT = get_free_port()
HOST = "127.0.0.1"
APP_URL = f"http://{HOST}:{SELECTED_PORT}"

# Development mode path resolution
if not getattr(sys, "frozen", False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(base_dir, "..", "src"))


# --- 3. Background Shiny Server ---
def start_server() -> None:
    """Runs the Shiny application server."""
    run_app(
        app, port=SELECTED_PORT, host=HOST, launch_browser=False, reload=False
    )


def wait_for_server(url: str, timeout: float = 20.0) -> bool:
    """Polls the server URL until it responds or the timeout is reached."""
    start_time = time.time()
    # Configure an opener that explicitly ignores proxy settings
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)

    while time.time() - start_time < timeout:
        try:
            with opener.open(url, timeout=1.0) as response:
                if response.status in (200, 302, 404):
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            # Server is still starting up; sleep briefly and retry
            time.sleep(0.3)
    return False


# --- 4. Fallback Desktop Window for Windows ---
def launch_fallback_window(url: str) -> None:
    """Launches Microsoft Edge in dedicated '--app' mode without tabs/URL bar."""
    edge_paths = [
        os.path.expandvars(
            r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
        ),
        os.path.expandvars(
            r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
        ),
    ]

    for edge_exe in edge_paths:
        if os.path.exists(edge_exe):
            try:
                proc = subprocess.Popen([edge_exe, f"--app={url}"])
                proc.wait()
                return
            except (OSError, subprocess.SubprocessError) as exc:
                logger.debug(
                    "Unable to launch Edge from %s: %s. Trying next option.",
                    edge_exe,
                    exc,
                )

    webbrowser.open(url)
    while True:
        time.sleep(1)


# --- 5. Application Startup ---
if __name__ == "__main__":
    multiprocessing.freeze_support()

    if os.environ.get("DIGIQUAL_IS_SPAWNED") == "1":
        pass
    else:
        os.environ["DIGIQUAL_IS_SPAWNED"] = "1"

        # 1. Start the Shiny server in a background daemon thread
        server_thread = threading.Thread(target=start_server)
        server_thread.daemon = True
        server_thread.start()

        # 2. Block until the server is actually answering requests
        server_ready = wait_for_server(APP_URL, timeout=20.0)
        if not server_ready:
            logger.warning(
                "Server did not respond within timeout; opening window anyway "
                "(it may still be starting up)."
            )

        # 3. Attempt to launch the webview window
        try:
            import webview
            from webview.menu import Menu, MenuAction, MenuSeparator

            webview.settings["ALLOW_DOWNLOADS"] = True

            def show_about() -> None:
                if webview.windows:
                    webview.windows[0].evaluate_js(
                        'alert("DigiQual\\nVersion 0.25.1\\nStatistical Toolkit for Reliability Assessment in NDT");'
                    )

            def open_documentation() -> None:
                if webview.windows:
                    webview.windows[0].evaluate_js(
                        'window.open("https://jgibristol.github.io/digiqual/", "_blank");'
                    )

            menu_items = [
                Menu(
                    "Help",
                    [
                        MenuAction("View Documentation", open_documentation),
                        MenuSeparator(),
                        MenuAction("About DigiQual", show_about),
                    ],
                )
            ]

            window = webview.create_window(
                "DigiQual", APP_URL, width=1200, height=800, resizable=True
            )

            gui_backend = "edgechromium" if sys.platform == "win32" else None
            webview.start(gui=gui_backend, private_mode=False, menu=menu_items)

        except (ImportError, RuntimeError, OSError) as exc:
            logger.exception("Webview failed to start: %s", exc)
            if sys.platform == "win32":
                logger.warning("Falling back to Edge app window.")
                launch_fallback_window(APP_URL)
            else:
                raise
