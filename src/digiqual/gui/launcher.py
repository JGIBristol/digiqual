"""Desktop launcher for the DigiQual Shiny app.

Starts the Shiny server on a free local port in a background thread and shows
it in a native pywebview window (falling back to an Edge ``--app`` window, then
the default browser).

Used by ``python -m digiqual.gui``, ``digiqual.dq_ui()`` and the Briefcase
desktop bundle (``app/src/digiqual_desktop``). Module-level imports are kept
light on purpose: packaged multiprocessing workers re-enter through ``main()``
and must not pay for importing shiny/pywebview.
"""

import logging
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

HOST = "127.0.0.1"
DOCS_URL = "https://jgibristol.github.io/digiqual/"

logger = logging.getLogger("digiqual.gui")


# --- 1. Packaged-app plumbing ---
def _handle_multiprocessing_child() -> None:
    """Runs this process as a multiprocessing helper if that's what it is.

    In a packaged app ``sys.executable`` is the app itself, so the workers that
    joblib's ``backend="multiprocessing"`` spawns (parallel bootstrap) launch
    the app again. With ``sys.frozen`` set (the Briefcase entry point sets it),
    multiprocessing spawns ``<exe> --multiprocessing-fork ...``; this routes
    that, and the POSIX resource tracker's ``<exe> -c ...``, to the right code
    instead of opening a second window. Mirrors PyInstaller's multiprocessing
    runtime hook. The self-test exercises all of this in CI.
    """
    if not getattr(sys, "frozen", False):
        return

    from multiprocessing import spawn

    # Worker process: runs the task and calls sys.exit() itself.
    spawn.freeze_support()

    # Resource trackers are started as `<exe> -c "from ... import main; ..."`
    # (joblib's loky one on every OS, including for backend="multiprocessing").
    argv = sys.argv
    if len(argv) >= 2 and argv[-2] == "-c" and argv[-1].startswith(
        (
            "from multiprocessing.resource_tracker import main",
            "from multiprocessing.forkserver import main",
            "from joblib.externals.loky.backend.resource_tracker import main",
        )
    ):
        exec(argv[-1])  # noqa: S102 - only the fixed prefixes matched above
        sys.exit()


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


def _setup_logging() -> Path:
    """Logs to a fixed, discoverable file and gives windowed builds real stdio.

    A GUI-subsystem executable has ``sys.stdout``/``sys.stderr`` set to
    ``None``; uvicorn's log formatter calls ``.isatty()`` on them and would
    kill the server thread, leaving a blank window. Pointing them at the log
    file fixes that and captures print() progress output too.
    """
    log_path = _resolve_log_path()
    # Deliberately left open: it backs logging and stdio for the whole run.
    log_stream = open(log_path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = log_stream
    if sys.stderr is None:
        sys.stderr = log_stream
    logging.basicConfig(
        stream=log_stream,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return log_path


# --- 2. Server helpers ---
def get_free_port() -> int:
    """Finds an available local TCP port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def start_server_thread(port: int) -> threading.Thread:
    """Runs the Shiny application server in a background daemon thread."""
    from shiny import run_app

    from digiqual.gui.app import app

    def _serve() -> None:
        try:
            run_app(app, port=port, host=HOST, launch_browser=False, reload=False)
        except Exception:
            logger.exception("Shiny server crashed")

    thread = threading.Thread(target=_serve, name="shiny-server", daemon=True)
    thread.start()
    return thread


def wait_for_server(url: str, timeout: float = 20.0) -> bool:
    """Polls the server URL until it responds or the timeout is reached."""
    # Explicitly ignore proxy settings: university/corporate proxies can't
    # reach the sandboxed loopback server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with opener.open(url, timeout=1.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.3)
    return False


# --- 3. Windows ---
def launch_webview_window(url: str) -> None:
    """Shows the app in a native pywebview window (blocks until closed)."""
    import webview
    from webview.menu import Menu, MenuAction, MenuSeparator

    from digiqual import __version__

    webview.settings["ALLOW_DOWNLOADS"] = True

    def show_about() -> None:
        if webview.windows:
            webview.windows[0].evaluate_js(
                f'alert("DigiQual\\nVersion {__version__}\\n'
                'Statistical Toolkit for Reliability Assessment in NDT");'
            )

    def open_documentation() -> None:
        webbrowser.open(DOCS_URL)

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

    webview.create_window("DigiQual", url, width=1200, height=800, resizable=True)
    gui_backend = "edgechromium" if sys.platform == "win32" else None
    webview.start(gui=gui_backend, private_mode=False, menu=menu_items)


def launch_fallback_window(url: str) -> None:
    """Shows the app in an Edge '--app' window, else the default browser.

    Blocks until the window closes, so the server thread stays alive.
    """
    if sys.platform == "win32":
        edge_paths = [
            os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        ]
        # A dedicated profile forces a new Edge process; otherwise msedge hands
        # the URL to an already-running Edge and exits immediately, and the
        # app would shut down underneath its own window.
        profile_dir = _resolve_log_path().parent / "edge-profile"
        for edge_exe in edge_paths:
            if os.path.exists(edge_exe):
                try:
                    subprocess.Popen(
                        [edge_exe, f"--app={url}", f"--user-data-dir={profile_dir}"]
                    ).wait()
                    return
                except (OSError, subprocess.SubprocessError) as exc:
                    logger.warning("Unable to launch Edge from %s: %s", edge_exe, exc)

    webbrowser.open(url)
    while True:
        time.sleep(1)


# --- 4. Self-test (used by CI against the packaged app) ---
def run_self_test() -> int:
    """Checks the C++ extension, process-based parallelism and the server."""
    from joblib import Parallel, delayed

    from digiqual.cpp_fallback import HAS_CPP

    ok = True
    logger.info("self-test: C++ extension available: %s", HAS_CPP)
    ok &= HAS_CPP

    try:
        squares = Parallel(n_jobs=2, backend="multiprocessing")(
            delayed(pow)(i, 2) for i in range(4)
        )
        parallel_ok = squares == [0, 1, 4, 9]
    except Exception:
        logger.exception("self-test: multiprocessing failed")
        parallel_ok = False
    logger.info("self-test: multiprocessing workers: %s", parallel_ok)
    ok &= parallel_ok

    port = get_free_port()
    start_server_thread(port)
    server_ok = wait_for_server(f"http://{HOST}:{port}", timeout=60.0)
    logger.info("self-test: HTTP server responded: %s", server_ok)
    ok &= server_ok

    logger.info("self-test: %s", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# --- 5. Entry point ---
def main() -> None:
    _handle_multiprocessing_child()

    log_path = _setup_logging()
    logger.info("Starting DigiQual (log: %s)", log_path)

    # Ensure local connections bypass any corporate/university proxy servers
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"

    if "--self-test" in sys.argv:
        code = run_self_test()
        logging.shutdown()
        # os._exit: don't wait on the daemon server thread or joblib cleanup.
        os._exit(code)

    port = get_free_port()
    url = f"http://{HOST}:{port}"
    start_server_thread(port)
    if not wait_for_server(url):
        logger.warning("Server did not respond within timeout; opening window anyway.")

    try:
        launch_webview_window(url)
    except Exception:
        # pywebview raises its own WebViewException (e.g. no WebView2 runtime),
        # and pythonnet/clr_loader failures vary, so catch broadly here.
        logger.exception("Webview failed to start; falling back.")
        launch_fallback_window(url)


if __name__ == "__main__":
    main()
