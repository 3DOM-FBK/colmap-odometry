"""Pluggable local features.

See :mod:`colmap_odometry.features.base` for the backend interfaces and
:mod:`colmap_odometry.features.registry` for how backends are discovered.
"""

from .base import Extractor, Features, JointModel, Matcher
from .frontend import FeatureFrontend, ResizeConfig, build_frontend
from .registry import backends, register

__all__ = [
    "Extractor",
    "FeatureFrontend",
    "Features",
    "JointModel",
    "Matcher",
    "ResizeConfig",
    "backends",
    "build_frontend",
    "register",
]
