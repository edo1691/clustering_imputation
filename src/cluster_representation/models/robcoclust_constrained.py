"""Compatibility exports for the custom robust co-clustering implementation."""

from .custom import robcoclust_constrained as _implementation

__all__ = [name for name in vars(_implementation) if not name.startswith("_")]


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
