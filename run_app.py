import sys
import os
import threading
import socket
import webview
from shiny import run_app
from digiqual.gui import app

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
    sys.path.insert(0, os.path.join(base_dir, "src"))

# --- 3. The Python API Bridge ---
class DownloadApi:
    def save_csv(self, content, filename):
        """
        Saves content to a file, using the filename provided by JS.
        """
        # Open the native macOS Save Dialog with the specific filename
        save_path = webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG,
            directory='/',
            save_filename=filename  # <--- Dynamic Name Here
        )

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception as e:
                print(f"Error saving file: {e}")

# --- 4. The JavaScript Injector ---
def inject_js(window):
    script = """
    // Map Button IDs to Filenames
    const filenameMap = {
        'download_lhs': 'experimental_design.csv',
        'download_new_samples_csv': 'refinement_samples.csv',
        'download_pod_results': 'pod_analysis_results.csv'
    };

    document.addEventListener('click', function(e) {
        // Find the download link
        var target = e.target.closest('a.shiny-download-link');

        if (target) {
            e.preventDefault();

            // 1. Determine the filename based on the ID
            // Default to 'results.csv' if ID is not in our map
            var name = filenameMap[target.id] || 'results.csv';

            // 2. Fetch the data
            fetch(target.href)
                .then(response => response.text())
                .then(data => {
                    // 3. Send data AND filename to Python
                    pywebview.api.save_csv(data, name);
                })
                .catch(err => alert('Download failed: ' + err));
        }
    });
    """
    window.evaluate_js(script)

def start_server():
    run_app(app, port=SELECTED_PORT, host=HOST, launch_browser=False, reload=False)

if __name__ == '__main__':
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()

    api = DownloadApi()
    window = webview.create_window(
        'DigiQual',
        f'http://{HOST}:{SELECTED_PORT}',
        width=1200,
        height=800,
        resizable=True,
        js_api=api
    )

    webview.start(func=inject_js, args=window, private_mode=False)
