__version__ = "0.10.0"

from .core import SimulationStudy
from . import pod
from . import diagnostics
from . import adaptive
from . import sampling
from . import plotting
from .gui import app as _shiny_app

def dq_ui(port=8000, launch_browser=True):
    """
    Launch the DigiQual Graphical User Interface (GUI).

    This function starts a local Shiny server and opens a windowed
    application for experimental design, diagnostics, and PoD analysis.

    Args
    ----------
    port : int, optional
        The port on which to run the local server, by default 8000.
    launch_browser : bool, optional
        Whether to automatically open the app in the default web browser,
        by default True.
    """
    from shiny import run_app

    # Ensure we use the package's internal path
    print(f"🚀 Launching DigiQual UI on http://localhost:{port}")
    run_app(_shiny_app, port=port, launch_browser=launch_browser)


__all__ = [
    "SimulationStudy",
    "pod",
    "diagnostics",
    "adaptive",
    "sampling",
    "plotting"
]
