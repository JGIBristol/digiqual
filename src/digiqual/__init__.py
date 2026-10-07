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
    User Interface for DigiQual Shiny Application
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
