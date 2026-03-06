__version__ = "0.12.3"

print(" Starting DigiQual... Loading statistical libraries (this may take a few seconds)...")

# Import core modules (telling Ruff to ignore the E402 rule for these specific lines)
from .core import SimulationStudy  # noqa: E402
from . import pod                  # noqa: E402
from . import diagnostics          # noqa: E402
from . import adaptive             # noqa: E402
from . import sampling             # noqa: E402
from . import plotting             # noqa: E402

def dq_ui():
    """
    User Interface for DigiQual Shiny Application
    """
    import sys
    import subprocess
    from pathlib import Path

    # 1. Define possible locations
    # Location A: Installed Package (e.g. site-packages/digiqual/app)
    # Location B: Local Dev Repo (e.g. Documents/DigiQual/app)

    current_dir = Path(__file__).parent

    possible_paths = [
        current_dir / "app" / "run_app.py",         # Installed Location
        current_dir.parent.parent / "app" / "run_app.py"  # Dev Location
    ]

    app_script = None
    for p in possible_paths:
        if p.exists():
            app_script = p
            break

    if app_script is None:
        print("❌ Critical Error: Could not find 'run_app.py'.")
        print(f"   Searched in: {[str(p) for p in possible_paths]}")
        return

    print(f"🚀 Launching DigiQual GUI from: {app_script}")

    # 2. Launch!
    # Using sys.executable ensures we use the active environment
    subprocess.Popen([sys.executable, str(app_script)], cwd=str(app_script.parent))

__all__ = [
    "SimulationStudy",
    "dq_ui",
    "pod",
    "diagnostics",
    "adaptive",
    "sampling",
    "plotting"
]
