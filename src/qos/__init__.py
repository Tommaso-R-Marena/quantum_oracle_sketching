"""Quantum Oracle Sketching (QOS) package."""

__version__ = "1.0.0"

from qos import config
from qos.core import oracle_sketch, sampling, state_sketch
from qos.data import generation
from qos.primitives import amplification, noise_model
from qos.qsvt import angles, transform
from qos.utils import numerical

__all__ = [
    "__version__",
    "config",
    "oracle_sketch",
    "sampling",
    "state_sketch",
    "generation",
    "amplification",
    "noise_model",
    "angles",
    "transform",
    "numerical",
]
