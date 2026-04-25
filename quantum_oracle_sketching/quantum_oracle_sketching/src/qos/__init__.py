"""Quantum Oracle Sketching (QOS): exponential quantum advantage in classical data processing.

This package implements quantum oracle sketching, a framework that enables
quantum computers to access classical data in superposition using only random
classical samples—without full-dataset memory overhead.

Key modules:
    core: Quantum state and oracle sketching implementations.
    qsvt: Quantum Singular Value Transform utilities.
    primitives: Quantum primitives such as amplitude amplification.
    utils: Numerical helpers, random generators, and block-encoding utilities.
    data: Data generation and sampling interfaces.
    experiments: Benchmarking and real-dataset evaluation tools.

Example:
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> from qos.core.state_sketch import q_state_sketch_flat
    >>> key = random.PRNGKey(0)
    >>> vector = jnp.ones(1024)  # flat vector
    >>> state, num_samples = q_state_sketch_flat(vector, unit_num_samples=10_000)

References:
    - Paper: arXiv:2604.07639
    - Blog: https://quantumfrontiers.com/2026/04/09/unleashing-the-advantage-of-quantum-ai/
"""

__version__ = "1.0.0"

from qos import config
from qos.core import oracle_sketch, sampling, state_sketch
from qos.data import generation
from qos.primitives import amplification
from qos.qsvt import angles, polynomial, transform
from qos.utils import encoding, matrices, numerical

__all__ = [
    "config",
    "oracle_sketch",
    "sampling",
    "state_sketch",
    "generation",
    "amplification",
    "angles",
    "polynomial",
    "transform",
    "encoding",
    "matrices",
    "numerical",
    "__version__",
]
