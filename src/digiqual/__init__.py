__version__ = "0.26.2"

# print(" Starting DigiQual... Loading statistical libraries (this may take a few seconds)...")

# Import core modules (telling Ruff to ignore the E402 rule for these specific lines)
from .core import SimulationStudy  # noqa: E402
from . import pod                  # noqa: E402
from . import diagnostics          # noqa: E402
from . import adaptive             # noqa: E402
from . import sampling             # noqa: E402
from . import plotting             # noqa: E402
from . import integration          # noqa: E402
from . import executors            # noqa: E402
from . import ahat                 # noqa: E402

def dq_ui():
    """
    Launch the DigiQual graphical user interface.

    Starts the app (``python -m digiqual.gui``) in a separate process, using
    the current Python environment, and returns immediately so the calling
    script or notebook stays usable. The app opens in its own desktop window,
    falling back to the default web browser if a native window can't be shown.

    The app's code lives in the ``digiqual.gui`` subpackage; see the Desktop
    App Architecture page of the documentation for how it is structured.
    """
    import subprocess
    import sys

    print("🚀 Launching DigiQual GUI...")
    # sys.executable ensures the GUI runs in the active environment
    subprocess.Popen([sys.executable, "-m", "digiqual.gui"])

__all__ = [
    "SimulationStudy",
    "dq_ui",
    "pod",
    "diagnostics",
    "adaptive",
    "sampling",
    "plotting",
    "integration",
    "executors",
    "ahat"
]
