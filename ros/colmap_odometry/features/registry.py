"""Discovery and construction of feature backends.

A backend is a class registered under a name with :func:`register`.  The
built-in backends live in :mod:`colmap_odometry.features.plugins`; every
module of that package is imported on discovery, so adding a backend means
adding one small module.  External packages can add backends through the
``colmap_odometry.features`` entry-point group (the entry point should
import the module that calls :func:`register`).

Plugin modules must import their heavy dependencies inside the backend
class, never at module level: discovery imports every plugin, while only the
selected backend may require its framework to be installed.  ``requires``
lists the importable modules a backend needs; unavailable backends are
reported instead of failing at import time.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Literal

Kind = Literal["extractor", "matcher", "joint"]
ENTRY_POINT_GROUP = "colmap_odometry.features"


@dataclass(frozen=True)
class Backend:
    name: str
    kind: Kind
    factory: Callable[..., Any]
    requires: tuple[str, ...]
    #: Priority within ``auto`` selection; higher wins, None excludes.
    auto_priority: int | None = None
    #: For extractors: matcher used when the config names none.
    default_matcher: str | None = None

    def missing_requirements(self) -> list[str]:
        return [module for module in self.requires
                if importlib.util.find_spec(module) is None]

    @property
    def available(self) -> bool:
        return not self.missing_requirements()


_REGISTRY: dict[tuple[Kind, str], Backend] = {}
_DISCOVERED = False


def register(
    kind: Kind,
    name: str,
    *,
    requires: tuple[str, ...] = (),
    auto_priority: int | None = None,
    default_matcher: str | None = None,
):
    """Class decorator registering a feature backend."""

    def decorator(factory):
        key = (kind, name)
        existing = _REGISTRY.get(key)
        if existing is not None and existing.factory is not factory:
            raise ValueError(f"feature {kind} '{name}' is registered twice")
        _REGISTRY[key] = Backend(name, kind, factory, tuple(requires),
                                 auto_priority, default_matcher)
        return factory

    return decorator


def discover() -> None:
    """Import built-in plugin modules and entry points once."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from . import plugins

    for module in pkgutil.iter_modules(plugins.__path__):
        importlib.import_module(f"{plugins.__name__}.{module.name}")
    try:
        entry_points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # Python < 3.10 API
        entry_points = metadata.entry_points().get(ENTRY_POINT_GROUP, [])
    for entry_point in entry_points:
        entry_point.load()


def backends(kind: Kind | None = None) -> list[Backend]:
    discover()
    return [backend for (backend_kind, _), backend in sorted(_REGISTRY.items())
            if kind is None or backend_kind == kind]


def get(kind: Kind, name: str) -> Backend:
    discover()
    backend = _REGISTRY.get((kind, name))
    if backend is None:
        known = ", ".join(item.name for item in backends(kind)) or "none"
        raise ValueError(f"unknown feature {kind} '{name}' (registered: {known})")
    missing = backend.missing_requirements()
    if missing:
        raise RuntimeError(
            f"feature {kind} '{name}' needs the Python modules {missing}, which are not "
            "importable (vendored libraries: run 'git submodule update --init')")
    return backend


def instantiate(backend: Backend, options: dict[str, Any]) -> Any:
    """Construct a backend, rejecting options its constructor does not take."""
    signature = inspect.signature(backend.factory)
    accepts_any = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                      for parameter in signature.parameters.values())
    if not accepts_any:
        unknown = sorted(set(options) - set(signature.parameters))
        if unknown:
            raise ValueError(
                f"feature {backend.kind} '{backend.name}' does not accept "
                f"options {unknown}; valid options: {sorted(signature.parameters)}")
    return backend.factory(**options)


def best_auto(kind: Kind) -> Backend | None:
    """Return the available backend with the highest ``auto_priority``."""
    candidates = [backend for backend in backends(kind)
                  if backend.auto_priority is not None and backend.available]
    return max(candidates, key=lambda backend: backend.auto_priority, default=None)
