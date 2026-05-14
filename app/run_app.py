import sys
import os
import threading
import socket
import webview
import multiprocessing
from webview.menu import Menu, MenuAction, MenuSeparator
from shiny import run_app
from app import app

# --- 1. Set Working Directory ---
if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)
    sys.path.insert(0, sys._MEIPASS)

# --- 2. Port Helper ---
def get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

SELECTED_PORT = get_free_port()
HOST = '127.0.0.1'

# Dev mode pathing
if not getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(base_dir, "..", "src"))

# --- 3. Enable Native Downloads ---
webview.settings['ALLOW_DOWNLOADS'] = True

# --- 4. Menu Bar Actions ---
def show_about():
    """Triggers an alert box with version info inside the app."""
    if webview.windows:
        webview.windows[0].evaluate_js(
            'alert("DigiQual\\nVersion 0.20.3\\nStatistical Toolkit for Reliability Assessment in NDT");'
        )

def open_documentation():
    """Opens the Quarto docs in the user's default web browser."""
    if webview.windows:
        webview.windows[0].evaluate_js(
            'window.open("https://jgibristol.github.io/digiqual/", "_blank");'
        )

menu_items = [
    Menu('Help', [
        MenuAction('View Documentation', open_documentation),
        MenuSeparator(),
        MenuAction('About DigiQual', show_about)
    ])
]

# --- 5. Application Startup ---
def start_server():
    run_app(app, port=SELECTED_PORT, host=HOST, launch_browser=False, reload=False)

if __name__ == '__main__':
    # 1. Standard PyInstaller intercept for workers
    multiprocessing.freeze_support()

    # 2. THE ULTIMATE FAILSAFE: Environment Variable Inheritance
    # If a process sees this flag already exists, it knows it is a child and skips the GUI!
    if os.environ.get('DIGIQUAL_IS_SPAWNED') == '1':
        pass # We are a stray background process. Do nothing and let it quietly exit.
    else:
        # We are the true main process! Set the flag for any future children.
        os.environ['DIGIQUAL_IS_SPAWNED'] = '1'
        
        t = threading.Thread(target=start_server)
        t.daemon = True
        t.start()

        window = webview.create_window(
            'DigiQual',
            f'http://{HOST}:{SELECTED_PORT}',
            width=1200,
            height=800,
            resizable=True
        )

        webview.start(private_mode=False, menu=menu_items)