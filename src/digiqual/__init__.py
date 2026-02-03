__version__ = "0.6.2"

# 1. Import Core Components
from .core import SimulationStudy

# 2. Import Submodules (Crucial for quartodoc to find them!)
from . import sampling
from . import diagnostics
from . import adaptive
from . import pod
from . import plotting

# 3. Define Public API
__all__ = [
    "SimulationStudy",
    "sampling",
    "diagnostics",
    "adaptive",
    "pod",
    "plotting"
]
