"""Stable, component-specific random number generators."""

import hashlib
from typing import Union

import numpy as np


def component_seed(master_seed: int, component: str) -> int:
    """Derive a stable 32-bit seed without Python's process-randomized hash()."""
    if not component:
        raise ValueError("component must be a non-empty string")
    digest = hashlib.sha256(
        "{}:{}".format(master_seed, component).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def generator(master_seed: int, component: str) -> np.random.Generator:
    """Create an independent NumPy Generator for a named component."""
    return np.random.default_rng(component_seed(master_seed, component))
