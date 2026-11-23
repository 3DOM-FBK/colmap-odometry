"""Feature libraries vendored in ``colmap_odometry/thirdparty``.

Each library is a git submodule pinned to a known commit (``git submodule
update --init`` after cloning).  Plugins call :func:`use` at import time so
the vendored copy takes precedence over any installed package, and build
their models inside :func:`weights_directory` so downloaded weights are kept
in ``thirdparty/weights`` (not committed) instead of the user cache.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "thirdparty"
WEIGHTS = ROOT / "weights"

#: Import name -> directory to put on ``sys.path``.
LIBRARIES = {
    "lightglue": ROOT / "LightGlue",
    "loma": ROOT / "LoMa" / "src",
}


def use(module: str) -> None:
    """Make the vendored ``module`` importable, ahead of installed copies."""
    path = LIBRARIES[module]
    if (path / module / "__init__.py").is_file() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


@contextmanager
def weights_directory():
    """Store weights fetched by ``torch.hub`` in ``thirdparty/weights``."""
    import torch

    previous = torch.hub.get_dir()
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(WEIGHTS))
    try:
        yield
    finally:
        torch.hub.set_dir(previous)
