__version__ = "0.28.0"

import logging as _logging
import sys as _sys

# Progress messages go through the "digiqual" logger. By default they are printed
# to stdout (as plain messages, like print), so scripts and notebooks show them.
# Silence or redirect them with the standard logging API, e.g.
#     logging.getLogger("digiqual").setLevel(logging.WARNING)
_package_logger = _logging.getLogger(__name__)
if not _package_logger.handlers:
    _handler = _logging.StreamHandler(_sys.stdout)
    _handler.setFormatter(_logging.Formatter("%(message)s"))
    _package_logger.addHandler(_handler)
    # Respect a level the user set before importing digiqual
    if _package_logger.level == _logging.NOTSET:
        _package_logger.setLevel(_logging.INFO)
    # Don't also pass messages to the root logger (avoids printing them twice when
    # an application, like the desktop launcher, configures root logging)
    _package_logger.propagate = False

# Import core modules after the logger is configured (hence E402)
from . import (  # noqa: E402
    adaptive,
    ahat,
    diagnostics,
    executors,
    integration,
    plotting,
    pod,
    sampling,
)
from .core import SimulationStudy  # noqa: E402


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

    _package_logger.info("Launching DigiQual GUI...")
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
